#pragma once

// module
module::module(network* nn) : m_network(nn) { }
network* module::getNetwork() {
	return m_network;
}

// module_function
template<int N, typename S>
module_function<N, S>::module_function(network* nn, concurrency::extent<2> extent, std::array<tensor_view<2>, N> inputs)
	: module(nn), m_output(nn, tensor_type_transient, extent), m_inputs(inputs) { }

template<int N, typename S>
module_function<N, S>::module_function(network* nn, std::array<tensor_view<2>, N> inputs)
	: module(nn), m_output(nn, tensor_type_transient, S::extent(inputs)), m_inputs(inputs) { }
template<int N, typename S>
module_function<N, S>::module_function(network* nn, tensor_view<2> input)
	: module(nn), m_output(nn, tensor_type_transient, S::extent({ tensor_view<2>(input) })), m_inputs({ tensor_view<2>(input) }) { }
template<int N, typename S>
module_function<N, S>::module_function(network* nn, tensor_view<2> input1, tensor_view<2> input2)
	: module(nn), m_output(nn, tensor_type_transient, S::extent({ tensor_view<2>(input1), tensor_view<2>(input2) })), m_inputs({ tensor_view<2>(input1), tensor_view<2>(input2) }) { }

template<int N, typename S>
void module_function<N, S>::updateOutput(state_provider const& stateProvider) {
	auto inputs = fixedMap(m_inputs, [&](auto& x) { return x.view(stateProvider); });
	table_view<2> output = m_output.view(stateProvider);
	auto state = static_cast<S*>(this)->getState(stateProvider);

	try {
		// Update outputs
		output.m_value.discard_data();

		parallel_for_each(
			output.extent(),
			[=](index<2> idx) restrict(amp) {
				S::forward(idx, output, inputs, state);
			}
		);

		if (m_network->getIsLearning()) {
			// Clear gradients
			for (int i = 0; i < N; ++i)
				inputs[i].m_gradient.discard_data();

			parallel_for_each(
				output.extent(),
					[=](index<2> idx) restrict(amp) {
					S::clearGradient(idx, output, inputs, state);
				}
			);
		}

	} catch (concurrency::runtime_exception& ex) {
		OutputDebugStringA(ex.what());
		DebugBreak();
	}
}
template<int N, typename S>
void module_function<N, S>::updateGradInput(state_provider const& stateProvider) {
	auto inputs = fixedMap(m_inputs, [&](auto& x) { return x.view(stateProvider); });
	table_view<2> output = m_output.view(stateProvider);
	auto state = static_cast<S*>(this)->getState(stateProvider);

	try {
		parallel_for_each(
			output.extent(),
			[=](index<2> idx) restrict(amp) {
				S::backward(idx, output, inputs, state);
			}
		);
	} catch (concurrency::runtime_exception& ex) {
		OutputDebugStringA(ex.what());
		DebugBreak();
	}
}

template<int N, typename S>
tensor_view<2> module_function<N, S>::getOutput() {
	return tensor_view<2>(m_output);
}
template<int N, typename S>
typename module_function<N, S>::state_t module_function<N, S>::getState(state_provider const& stateProvider) {
	return state_t();
};
template<int N, typename S>
void module_function<N, S>::clearGradient(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, N> const& inputs, state_t state) restrict(amp) {
	range_to<N>::each<0>([=](int i, auto& inputi) restrict(amp) {
		inputi.m_gradient[idx] = 0.0f;
	}, inputs);
}
template<int N, typename S>
extent<2> module_function<N, S>::extent(std::array<tensor_view<2>, N> inputs) {
	auto extent = inputs.begin()->extent();
	for (auto& input : inputs) {
		if (input.extent() != extent)
			throw "Extent mismatch";
	}
	return extent;
}

// module_scalar
template<int N, typename S>
template<bool unused>
void module_scalar<N, S>::forward(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, N> const& inputs, state_t state) restrict(amp) {
	// Default implementation of `forward` is essentially `foldl <op> [inputs...]`

	float acc = S::identity();
	range_to<N>::each<uniform_param, 0>([=](int i, float& acc, auto& inputi) restrict(amp) {
		acc = S::op(acc, inputi.m_value[idx]);
	}, acc, inputs);
	output.m_value[idx] = acc;
}
template<int N, typename S>
template<bool unused>
void module_scalar<N, S>::backward(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, N> const& inputs, state_t state) restrict(amp) {
	// This is a fun little snippet of code which calculates all N partial
	// derivatives of `foldl <op> [inputs...]` in O(N) time

	// We can't use normal loops because AMP is funny about them...

	float x[N];
	range_to<N>::each<0, 0>([=](int i, float& xi, auto& inputi) restrict(amp) {
		xi = inputi.m_value[idx];
	}, x, inputs);

	float acc[N];
	acc[0] = S::identity();
	range_to<N - 1>::each<1, 0, 0>([=](int i, float& acci1, float& acci, auto& xi) restrict(amp) {
		acci1 = S::op(acci, xi);
	}, acc, acc, x);

	float dacc0[N];
	dacc0[N - 1] = 1.0f;
	range_to<N - 1>::eachRev<0, 1, 1>([=](int i, float& dacc0i, float& acci, auto& xi) restrict(amp) {
		dacc0i = S::dop0(acci, xi);
	}, dacc0, acc, x);

	float y[N];
	range_to<N - 1>::eachRev<0, 0, 0, 0>([=](int i, float& yi, float& acci, auto& xi, float& dacc0i) restrict(amp) {
		float dacc1 = S::dop1(acci, xi);
		yi = dacc1*dacc0i;
	}, y, acc, x, dacc0);

	float gradient = output.m_gradient[idx];
	range_to<N>::each<0, 0>([=](int i, auto& inputi, float& yi) restrict(amp) {
		inputi.m_gradient[idx] += gradient*yi;
	}, inputs, y);
}

// module_unary
template<typename S>
module_unary<S>::module_unary(network* nn, tensor_view<2> input) : module_scalar(nn, make_array(input)) { }

template<typename S>
void module_unary<S>::forward(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp) {
	output.m_value[idx] = S::op(inputs[0].m_value[idx]);
}
template<typename S>
void module_unary<S>::backward(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp) {
	inputs[0].m_gradient[idx] += output.m_gradient[idx] * S::dop(output.m_value[idx], inputs[0].m_value[idx]);
}

// module_binary
template<typename S>
module_binary<S>::module_binary(network* nn, tensor_view<2> input1, tensor_view<2> input2) : module_scalar(nn, make_array(input1, input2)) { }

// module_add
template<int N>
float module_add<N>::identity() restrict(amp) {
	return 0.0f;
}
template<int N>
float module_add<N>::op(float a, float b) restrict(amp) {
	return a + b;
}
template<int N>
float module_add<N>::dop0(float a, float b) restrict(amp) {
	return 1.0f;
}
template<int N>
float module_add<N>::dop1(float a, float b) restrict(amp) {
	return 1.0f;
}

// module_sub
float module_sub::identity() restrict(amp) {
	return 0.0f;
}
float module_sub::op(float a, float b) restrict(amp) {
	return -(a + b);
}
float module_sub::dop0(float a, float b) restrict(amp) {
	return -1.0f;
}
float module_sub::dop1(float a, float b) restrict(amp) {
	return -1.0f;
}

// module_neg
float module_neg::op(float input) restrict(amp) {
	return -input;
}
float module_neg::dop(float output, float input) restrict(amp) {
	return -1.0f;
}

// module_mul
template<int N>
float module_mul<N>::identity() restrict(amp) {
	return 1.0f;
}
template<int N>
float module_mul<N>::op(float a, float b) restrict(amp) {
	return a*b;
}
template<int N>
float module_mul<N>::dop0(float a, float b) restrict(amp) {
	return b;
}
template<int N>
float module_mul<N>::dop1(float a, float b) restrict(amp) {
	return a;
}

// module_div
float module_div::identity() restrict(amp) {
	return 1.0f;
}
float module_div::op(float a, float b) restrict(amp) {
	return 1.0f / (a*b);
}
float module_div::dop0(float a, float b) restrict(amp) {
	return -1.0f / (a*a*b);
}
float module_div::dop1(float a, float b) restrict(amp) {
	return -1.0f / (a*b*b);
}

// module_rcp
float module_rcp::op(float input) restrict(amp) {
	return 1.0f / input;
}
float module_rcp::dop(float output, float input) restrict(amp) {
	return -1.0f / (input*input);
}

// module_sigmoid
float module_sigmoid::op(float input) restrict(amp) {
	return 1.0f / (1.0f + concurrency::fast_math::exp(-input));
}
float module_sigmoid::dop(float output, float input) restrict(amp) {
	return output*(1.0f - output);
}

// module_tanh
float module_tanh::op(float input) restrict(amp) {
	return concurrency::fast_math::tanh(input);
}
float module_tanh::dop(float output, float input) restrict(amp) {
	return 1.0f - output*output;
}

// module_linear
void module_linear::updateGradInput(state_provider const& stateProvider) {
	// The gradients need to be calculated in the other direction from the outputs
	auto inputs = fixedMap(m_inputs, [&](auto& x) { return x.view(stateProvider); });
	table_view<2> output = m_output.view(stateProvider);
	auto state = getState(stateProvider);

	try {
		parallel_for_each(
			state.bias.extent(),
			[=](index<1> idx) restrict(amp) {
				backwardBias(idx, output, inputs, state);
			}
		);
		parallel_for_each(
			state.weights.extent(),
			[=](index<2> idx) restrict(amp) {
				backwardWeights(idx, output, inputs, state);
			}
		);
		parallel_for_each(
			m_inputs[0].extent(),
			[=](index<2> idx) restrict(amp) {
				backward(idx, output, inputs, state);
			}
		);
	} catch (concurrency::runtime_exception& ex) {
		OutputDebugStringA(ex.what());
		DebugBreak();
	}
}

module_linear::module_linear(network* nn, concurrency::extent<1> extent, tensor_view<2> input)
	: module_function(nn, concurrency::extent<2>(input.extent()[0], extent[0]), { input })
	, m_weights(nn, tensor_type_weight, concurrency::extent<2>(extent[0], module_function::extent({ input })[0]))
	, m_bias(nn, tensor_type_weight, extent) { }

void module_linear::forward(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp) {
	table_view<2> input = inputs[0];

	// Initial bias
	float acc = state.bias.m_value[idx[1]];

	// Matrix-vector multiplication
	for (int i = 0; i < state.weights.m_value.extent[1]; ++i) {
		index<2> idx2(idx[1], i);
		index<2> idx3(idx[0], i);
		acc += state.weights.m_value[idx2] * input.m_value[idx3];
	}
	output.m_value[idx] = acc;
}
void module_linear::backwardBias(index<1> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp) {
	float gradient = 0.0f;
	for (int i = 0; i < output.extent()[0]; ++i) {
		index<2> idx2(i, idx[0]);
		gradient += output.m_gradient[idx2];
	}
	state.bias.m_gradient[idx] += gradient;
}
void module_linear::backwardWeights(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp) {
	table_view<2> input = inputs[0];

	float acc = 0.0f;

	// Matrix-vector multiplication
	for (int i = 0; i < output.m_gradient.extent[0]; ++i) {
		index<2> idx2(i, idx[0]);
		index<2> idx3(i, idx[1]);

		float gradient = output.m_gradient[idx2];
		float value = input.m_value[idx3];

		// Update weight gradient
		acc += gradient * value;
	}
	state.weights.m_gradient[idx] = acc;
}
void module_linear::backward(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp) {
	table_view<2> input = inputs[0];

	float acc = 0.0f;
	float value = input.m_value[idx];

	// Matrix-vector multiplication
	for (int i = 0; i < output.m_gradient.extent[1]; ++i) {
		index<2> idx2(idx[0], i);
		index<2> idx3(i, idx[1]);

		float gradient = output.m_gradient[idx2];

		acc += state.weights.m_value[idx3] * gradient;
	}
	input.m_gradient[idx] = acc;
}
void module_linear::clearGradient(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp) {
	module_function::clearGradient(idx, output, inputs, module_function::state_t());
}

module_linear::state_t module_linear::getState(state_provider const& stateProvider) {
	return state_t{ m_weights.view(stateProvider), m_bias.view(stateProvider) };
}

// module_log_soft_max
module_log_soft_max::module_log_soft_max(network* nn, tensor_view<2> input)
	: module(nn), m_output(nn, tensor_type_transient, input.extent()), m_scratch(nn, tensor_type_transient, input.extent()), m_input(input), m_maxInput(nn, tensor_type_transient, extent<1>(input.extent()[0])) { }

void module_log_soft_max::updateOutput(state_provider const& stateProvider) {
	table_view<2> input = m_input.view(stateProvider);
	table_view<2> scratch = m_scratch.view(stateProvider);
	table_view<2> output = m_output.view(stateProvider);
	table_view<1> maxInput = m_maxInput.view(stateProvider);

	try {
		// Update outputs
		output.m_value.discard_data();
		scratch.m_value.discard_data();
		maxInput.m_value.discard_data();

		float negativeInfinity = -std::numeric_limits<float>::infinity();
		parallel_for_each(
			maxInput.extent(),
			[=](index<1> idx) restrict(amp) {
				float maxInput = reduce(input.m_value[idx[0]], negativeInfinity, [=](float a, float b) restrict(amp, cpu) {
					return (a > b) ? a : b;
				});
			}
		);

		parallel_for_each(
			output.extent(),
			[=](index<2> idx) restrict(amp) {
				scratch.m_value[idx] = fast_math::expf(input.m_value[idx] - maxInput.m_value[idx[0]]);
			}
		);

		parallel_for_each(
			maxInput.extent(),
			[=](index<1> idx) restrict(amp) {
				float logSum = reduce(scratch.m_value[idx[0]], 0.0f, [=](float a, float b) restrict(amp, cpu) {
					return a + b;
				});
				maxInput.m_value[idx] += fast_math::logf(logSum);
			}
		);


		parallel_for_each(
			output.extent(),
			[=](index<2> idx) restrict(amp) {
				output.m_value[idx] = input.m_value[idx] - maxInput.m_value[idx[0]];
			}
		);

		if (m_network->getIsLearning()) {
			// Clear gradients
			input.m_gradient.discard_data();
			scratch.m_gradient.discard_data();

			parallel_for_each(
				output.extent(),
				[=](index<2> idx) restrict(amp) {
					input.m_gradient[idx] = 0.0f;
					scratch.m_gradient[idx] = 0.0f;
				}
			);
		}

	} catch (concurrency::runtime_exception& ex) {
		OutputDebugStringA(ex.what());
		DebugBreak();
	}
}
void module_log_soft_max::updateGradInput(state_provider const& stateProvider) {
	table_view<2> input = m_input.view(stateProvider);
	table_view<2> scratch = m_scratch.view(stateProvider);
	table_view<2> output = m_output.view(stateProvider);
	table_view<1> maxInput = m_maxInput.view(stateProvider);

	try {
		maxInput.m_gradient.discard_data();
		parallel_for_each(
			maxInput.extent(),
			[=](index<1> idx) restrict(amp) {
				maxInput.m_gradient[idx] = reduce(output.m_gradient[idx[0]], 0.0f, [=](float a, float b) restrict(amp, cpu) {
					return a + b;
				});
			}
		);

		parallel_for_each(
			output.extent(),
			[=](index<2> idx) restrict(amp) {
				input.m_gradient[idx] += output.m_gradient[idx] - maxInput.m_gradient[idx[0]]*fast_math::expf(output.m_value[idx]);
			}
		);
	} catch (concurrency::runtime_exception& ex) {
		OutputDebugStringA(ex.what());
		DebugBreak();
	}
}

tensor_view<2> module_log_soft_max::getOutput() {
	return m_output;
}

// module_input
module_input::module_input(network* nn, extent<2> extent) : module(nn), m_output(nn, tensor_type_transient, extent) { }

void module_input::updateOutput(state_provider const& stateProvider) {
}
void module_input::updateGradInput(state_provider const& stateProvider) {
}

void module_input::setValue(state_provider const& stateProvider, array_view<float, 2> value) {
	if (value.extent != m_output.extent())
		throw "Extent mismatch";
	value.copy_to(m_output.view(stateProvider).m_value);
}
void module_input::getGradient(state_provider const& stateProvider, array_view<float, 2> gradient) {
	if (gradient.extent != m_output.extent())
		throw "Extent mismatch";
	m_output.view(stateProvider).m_gradient.copy_to(gradient);
}
tensor_view<2> module_input::getOutput() {
	return tensor_view<2>(m_output);
}

// module_class_embedding
module_class_embedding::module_class_embedding(network* nn, unsigned numClasses, extent<2> extent)
	: module(nn), m_output(nn, tensor_type_transient, extent), m_weights(nn, tensor_type_weight, concurrency::extent<2>(numClasses, extent[1])), m_input(nn, concurrency::extent<1>(extent[0])) { }

void module_class_embedding::updateOutput(state_provider const& stateProvider) {
	array_view<int, 1> input = m_input.view(stateProvider);
	table_view<2> output = m_output.view(stateProvider);
	table_view<2> weights = m_weights.view(stateProvider);
	
	try {
		parallel_for_each(
			output.extent(),
			[=](index<2> idx) restrict(amp) {
				int classIndex = input[idx[0]];
				if (classIndex >= 0) {
					index<2> idx2(classIndex, idx[1]);
					output.m_value[idx] = weights.m_value[idx2];
				} else {
					output.m_value[idx] = 0.0f;
				}
			}
		);
	} catch (concurrency::runtime_exception& ex) {
		OutputDebugStringA(ex.what());
		DebugBreak();
	}
}
void module_class_embedding::updateGradInput(state_provider const& stateProvider) {
	array_view<int, 1> input = m_input.view(stateProvider);
	table_view<2> output = m_output.view(stateProvider);
	table_view<2> weights = m_weights.view(stateProvider);

	try {
		parallel_for_each(
			weights.extent(),
			[=](index<2> idx) restrict(amp) {
				float acc = 0.0f;
				for (int i = 0; i < output.extent()[0]; ++i) {
					if (idx[0] == input[i]) {
						index<2> idx2(i, idx[1]);
						acc += output.m_gradient[idx2];
					}
				}
				
				weights.m_gradient[idx] += acc;
			}
		);
	} catch (concurrency::runtime_exception& ex) {
		OutputDebugStringA(ex.what());
		DebugBreak();
	}
}

void module_class_embedding::setInput(state_provider const& stateProvider, array_view<int, 1> classes) {
	if (classes.extent[0] != m_output.extent()[0])
		throw "Extent mismatch";
	classes.copy_to(m_input.view(stateProvider));
}
inline array_view<int, 1> module_class_embedding::getInput(state_provider const& stateProvider) {
	return m_input.view(stateProvider);
}
tensor_view<2> module_class_embedding::getOutput() {
	return tensor_view<2>(m_output);
}

// module_state
module_state::module_state(network* nn, concurrency::extent<2> extent) : module(nn), m_output(nn, tensor_type_state, extent), m_hasInput(false) { }

void module_state::updateOutput(state_provider const& stateProvider) {
	if (!m_hasInput)
		throw "State must have precisely one input";
}
void module_state::updateGradInput(state_provider const& stateProvider) {
}

tensor_view<2> module_state::getOutput() {
	return tensor_view<2>(m_output);
}
void module_state::setInput(tensor_view<2> input) {
	m_network->make<module_state_input>(this, input);
}

// module_state_input
inline module_state_input::module_state_input(network* nn, module_state* state, tensor_view<2> input) : module(nn), m_state(state->getOutput()), m_input(input) {
	if (nn != state->getNetwork())
		throw "Input to state must be within the same network";
	if (state->m_hasInput)
		throw "State can only have one input";
	state->m_hasInput = true;
}

void module_state_input::updateOutput(state_provider const& stateProvider) {
	table_view<2> input = m_input.view(stateProvider);

	// Propagate input to next state
	input.m_value.copy_to(m_state.nextView(stateProvider).m_value);

	if (m_network->getIsLearning()) {
		// Clear gradients
		input.m_gradient.discard_data();

		parallel_for_each(
			input.extent(),
			[=](index<2> idx) restrict(amp) {
				input.m_gradient[idx] = 0.0f;
			}
		);
	}
}
void module_state_input::updateGradInput(state_provider const& stateProvider) {
	// Propagate gradient back from next state
	table_view<2> input = m_input.view(stateProvider);
	table_view<2> nextState = m_state.nextView(stateProvider);

	try {
		parallel_for_each(
			nextState.extent(),
			[=](index<2> idx) restrict(amp) {
				input.m_gradient[idx] += nextState.m_gradient[idx];
			}
		);
	} catch (concurrency::runtime_exception& ex) {
		OutputDebugStringA(ex.what());
		DebugBreak();
	}
}

// module_container
template<int N, int M, typename S>
template<typename... P>
module_container<N, M, S>::module_container(network* nn, P&&... p) : module(nn), m_outputs(S::build(nn, std::forward<P>(p)...)) { }
template<int N, int M, typename S>
void module_container<N, M, S>::updateOutput(state_provider const& stateProvider) {
}
template<int N, int M, typename S>
void module_container<N, M, S>::updateGradInput(state_provider const& stateProvider) {
}

// module_container_function
template<int M, typename S>
tensor_view<M> module_container_function<M, S>::getOutput() {
	return m_outputs[0];
}

// module_lstm
std::array<tensor_view<2>, 1> module_lstm::build(network* nn, extent<1> extent, tensor_view<2> input) {
	concurrency::extent<2> stateExtent(input.extent()[0], extent[0]);

	auto internalState = nn->make<module_state>(stateExtent);
	auto outputState = nn->make<module_state>(stateExtent);

	// Prepare inputs for the four gates
	module_add<2>* inputs[4];
	for (int i = 0; i < 4; ++i) {
		// Feed in the current input, and the previous output
		auto inputA = nn->make<module_linear>(extent, input);
		auto inputB = nn->make<module_linear>(extent, outputState);
		inputs[i] = nn->make<module_add<2>>(inputA, inputB);
	}

	auto inGate = nn->make<module_sigmoid>(inputs[0]);
	auto forgetGate = nn->make<module_sigmoid>(inputs[1]);
	auto outGate = nn->make<module_sigmoid>(inputs[2]);
	auto inTransform = nn->make<module_tanh>(inputs[3]);

	auto nextValue = nn->make<module_add<>>(
		nn->make<module_mul<>>(internalState, forgetGate),
		nn->make<module_mul<>>(inGate, inTransform)
	);

	auto nextOutput = nn->make<module_mul<>>(
		outGate,
		nn->make<module_tanh>(nextValue)
	);

	internalState->setInput(nextValue);
	outputState->setInput(nextOutput);

	return{ nextOutput };
}

// module_softmax
std::array<tensor_view<2>, 1> module_softmax::build(network* nn, extent<1> extent, tensor_view<2> input) {
	return{
		nn->make<module_log_soft_max>(
			nn->make<module_linear>(extent, input)
		)
	};
}

// module_class_nll_criterion
module_class_nll_criterion::module_class_nll_criterion(network* nn, tensor_view<2> input)
	: module(nn), m_input(input), m_target(nn, concurrency::extent<1>(input.extent()[0])), m_output(nn, concurrency::extent<1>(input.extent()[0])) { }

void module_class_nll_criterion::updateOutput(state_provider const& stateProvider) {
	if (m_network->getIsLearning()) {
		table_view<2> input = m_input.view(stateProvider);
		array_view<int, 1> target = m_target.view(stateProvider);
		array_view<float, 1> output = m_output.view(stateProvider);

		try {
			float scale = 1.0f / output.extent[0];

			parallel_for_each(
				output.extent,
				[=](index<1> idx) restrict(amp) {
					int classIndex = target[idx];
					index<2> idx2(idx[0], classIndex);
					float loss = -input.m_value[idx2] * scale;
					output[idx] = loss;
				}
			);

			// Clear gradients
			input.m_gradient.discard_data();

			parallel_for_each(
				input.extent(),
				[=](index<2> idx) restrict(amp) {
					input.m_gradient[idx] = 0.0f;
				}
			);
		} catch (concurrency::runtime_exception& ex) {
			OutputDebugStringA(ex.what());
			DebugBreak();
		}
	}
}
void module_class_nll_criterion::updateGradInput(state_provider const& stateProvider) {
	table_view<2> input = m_input.view(stateProvider);
	array_view<int, 1> target = m_target.view(stateProvider);
	array_view<float, 1> output = m_output.view(stateProvider);

	try {
		float scale = 1.0f / output.extent[0];

		parallel_for_each(
			output.extent,
			[=](index<1> idx) restrict(amp) {
				int classIndex = target[idx];
				index<2> idx2(idx[0], classIndex);
				input.m_gradient[idx2] -= scale;
			}
		);
	} catch (concurrency::runtime_exception& ex) {
		OutputDebugStringA(ex.what());
		DebugBreak();
	}
}

void module_class_nll_criterion::setTarget(state_provider const& stateProvider, array_view<int, 1> classes) {
	if (classes.extent[0] != m_input.extent()[0])
		throw "Extent mismatch";
	classes.copy_to(m_target.view(stateProvider));
}
array_view<int, 1> module_class_nll_criterion::getTarget(state_provider const& stateProvider) {
	return m_target.view(stateProvider);
}
buffer_view<float> module_class_nll_criterion::getOutput() {
	return buffer_view<float>(m_output);
}
