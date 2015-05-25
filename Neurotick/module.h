#pragma once

class module {
protected:
	network* m_network;

	inline module(network* nn) : m_network(nn) { }

public:
	virtual void updateOutput() = 0;
	virtual void updateGradInput() = 0;
};

template<int N, typename S>
class module_function : public module {
protected:
	tensor<> m_output;
	std::array<tensor_view<>, N> m_inputs;

	inline module_function(network* nn, extent<1> extent, std::array<tensor_view<>, N> inputs) : module(nn), m_output(nn, tensor_type_transient, extent), m_inputs(inputs) { }

public:
	struct state_t {};

	inline module_function(network* nn, std::array<tensor_view<>, N> inputs) : module(nn), m_output(nn, tensor_type_transient, S::extent(inputs)), m_inputs(inputs) { }

	virtual void updateOutput() {
		auto inputs = fixedMap(m_inputs, [](auto& x) { return x.view(); });
		table_view<> output = m_output.view();
		auto state = static_cast<S*>(this)->getState();

		try {
			// Update outputs
			output.m_value.discard_data();

			parallel_for_each(
				output.extent(),
				[=](index<1> idx) restrict(amp) {
					S::forward(idx, output, inputs, state);
				}
			);

			if (m_network->getIsLearning()) {
				// Clear gradients
				for (int i = 0; i < N; ++i)
					inputs[i].m_gradient.discard_data();

				parallel_for_each(
					output.extent(),
					[=](index<1> idx) restrict(amp) {
						S::clearGradient(idx, output, inputs, state);
					}
				);
			}

		} catch (concurrency::runtime_exception& ex) {
			OutputDebugStringA(ex.what());
			DebugBreak();
		}
	}
	virtual void updateGradInput() {
		auto inputs = fixedMap(m_inputs, [](auto& x) { return x.view(); });
		table_view<> output = m_output.view();
		auto state = static_cast<S*>(this)->getState();

		try {
			parallel_for_each(
				output.extent(),
				[=](index<1> idx) restrict(amp) {
					S::backward(idx, output, inputs, state);
				}
			);
		} catch (concurrency::runtime_exception& ex) {
			OutputDebugStringA(ex.what());
			DebugBreak();
		}
	}

	inline tensor_view<> getOutput() {
		return tensor_view<>(m_output);
	}
	inline state_t getState() {
		return state_t();
	};
	static inline void clearGradient(index<1> idx, table_view<> const& output, fixed_array<table_view<>, N> const& inputs, state_t state) restrict(amp) {
		range_to<N>::each<0>([=](int i, auto& inputi) restrict(amp) {
			inputi.m_gradient[idx] = 0.0f;
		}, inputs);
	}
	static inline extent<1> extent(std::array<tensor_view<>, N> inputs) {
		auto extent = inputs.begin()->extent();
		for (auto& input : inputs) {
			if (input.extent() != extent)
				throw "Extent mismatch";
		}
		return extent;
	}
};

// Base class for modules of the form `foldl <op> [inputs...]`,
// which includes scalar arithmetic
template<int N, typename S>
class module_scalar : public module_function<N, S> {
public:
	using module_function::module_function;

	template<bool unused = false>
	static inline void forward(index<1> idx, table_view<> const& output, fixed_array<table_view<>, N> const& inputs, state_t state) restrict(amp) {
		// Default implementation of `forward` is essentially `foldl <op> [inputs...]`

		float acc = S::identity();
		range_to<N>::each<uniform_param, 0>([=](int i, float& acc, auto& inputi) restrict(amp) {
			acc = S::op(acc, inputi.m_value[idx]);
		}, acc, inputs);
		output.m_value[idx] = acc;
	}
	template<bool unused = false>
	static inline void backward(index<1> idx, table_view<> const& output, fixed_array<table_view<>, N> const& inputs, state_t state) restrict(amp) {
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
};

template<typename S>
class module_unary : public module_scalar<1, S> {
public:
	inline module_unary(network* nn, tensor_view<> input) : module_scalar(nn, make_array(input)){ }

	static inline void forward(index<1> idx, table_view<> const& output, fixed_array<table_view<>, 1> const& inputs, state_t state) restrict(amp) {
		output.m_value[idx] = S::op(inputs[0].m_value[idx]);
	}
	static inline void backward(index<1> idx, table_view<> const& output, fixed_array<table_view<>, 1> const& inputs, state_t state) restrict(amp) {
		inputs[0].m_gradient[idx] += output.m_gradient[idx] * S::dop(output.m_value[idx], inputs[0].m_value[idx]);
	}
};

template<typename S>
class module_binary : public module_scalar<2, S> {
public:
	inline module_binary(network* nn, tensor_view<> input1, tensor_view<> input2) : module_scalar(nn, make_array(input1, input2)) { }
};

template<int N = 2>
class module_add : public module_scalar<N, module_add<N>> {
public:
	using module_scalar::module_scalar;

	static inline float identity() restrict(amp) {
		return 0.0f;
	}
	static inline float op(float a, float b) restrict(amp) {
		return a + b;
	}
	static inline float dop0(float a, float b) restrict(amp) {
		return 1.0f;
	}
	static inline float dop1(float a, float b) restrict(amp) {
		return 1.0f;
	}
};

class module_sub : public module_binary<module_sub> {
public:
	using module_binary::module_binary;

	static inline float identity() restrict(amp) {
		return 0.0f;
	}
	static inline float op(float a, float b) restrict(amp) {
		return -(a + b);
	}
	static inline float dop0(float a, float b) restrict(amp) {
		return -1.0f;
	}
	static inline float dop1(float a, float b) restrict(amp) {
		return -1.0f;
	}
};
class module_neg : public module_unary<module_neg> {
public:
	using module_unary::module_unary;

	static inline float op(float input) restrict(amp) {
		return -input;
	}
	static inline float dop(float output, float input) restrict(amp) {
		return -1.0f;
	}
};

template<int N = 2>
class module_mul : public module_scalar<N, module_mul<N>> {
public:
	using module_scalar::module_scalar;

	static inline float identity() restrict(amp) {
		return 1.0f;
	}
	static inline float op(float a, float b) restrict(amp) {
		return a*b;
	}
	static inline float dop0(float a, float b) restrict(amp) {
		return b;
	}
	static inline float dop1(float a, float b) restrict(amp) {
		return a;
	}
};

class module_div : public module_binary<module_div> {
public:
	using module_binary::module_binary;

	static inline float identity() restrict(amp) {
		return 1.0f;
	}
	static inline float op(float a, float b) restrict(amp) {
		return 1.0f / (a*b);
	}
	static inline float dop0(float a, float b) restrict(amp) {
		return -1.0f / (a*a*b);
	}
	static inline float dop1(float a, float b) restrict(amp) {
		return -1.0f / (a*b*b);
	}
};
class module_rcp : public module_unary<module_rcp> {
public:
	using module_unary::module_unary;

	static inline float op(float input) restrict(amp) {
		return 1.0f / input;
	}
	static inline float dop(float output, float input) restrict(amp) {
		return -1.0f / (input*input);
	}
};
class module_sigmoid : public module_unary<module_sigmoid> {
public:
	using module_unary::module_unary;

	static inline float op(float input) restrict(amp) {
		return 1.0f / (1.0f + concurrency::fast_math::exp(-input));
	}
	static inline float dop(float output, float input) restrict(amp) {
		return output*(1.0f - output);
	}
};
class module_tanh : public module_unary<module_tanh> {
public:
	using module_unary::module_unary;

	static inline float op(float input) restrict(amp) {
		return concurrency::fast_math::tanh(input);
	}
	static inline float dop(float output, float input) restrict(amp) {
		return 1.0f - output*output;
	}
};
class module_linear : public module_function<1, module_linear> {
protected:
	tensor<2> m_weights;
	tensor<> m_bias;
public:
	struct state_t {
		table_view<2> weights;
		table_view<> bias;
	};

	// The gradients need to be calculated in the other direction from the outputs
	virtual void updateGradInput() {
		auto inputs = fixedMap(m_inputs, [](auto& x) { return x.view(); });
		table_view<> output = m_output.view();
		auto state = getState();

		try {
			parallel_for_each(
				m_output.extent(),
				[=](index<1> idx) restrict(amp) {
					backwardBias(idx, output, inputs, state);
				}
			);
			parallel_for_each(
				m_inputs[0].extent(),
				[=](index<1> idx) restrict(amp) {
					backward(idx, output, inputs, state);
				}
			);
		} catch (concurrency::runtime_exception& ex) {
			OutputDebugStringA(ex.what());
			DebugBreak();
		}
	}

	inline module_linear(network* nn, concurrency::extent<1> extent, tensor_view<> input)
		: module_function(nn, extent, { input })
		, m_weights(nn, tensor_type_weight, concurrency::extent<2>(extent[0], module_function::extent({ input })[0]))
		, m_bias(nn, tensor_type_weight, extent) { }

	static inline void forward(index<1> idx, table_view<> const& output, fixed_array<table_view<>, 1> const& inputs, state_t state) restrict(amp) {
		table_view<> input = inputs[0];

		// Initial bias
		float acc = state.bias.m_value[idx];

		// Matrix-vector multiplication
		for (int i = 0; i < state.weights.m_value.extent[1]; ++i) {
			index<2> idx2(idx[0], i);
			acc += state.weights.m_value[idx2] * input.m_value[i];
		}
		output.m_value[idx] = acc;
	}
	static inline void backwardBias(index<1> idx, table_view<> const& output, fixed_array<table_view<>, 1> const& inputs, state_t state) restrict(amp) {
		state.bias.m_gradient[idx] += output.m_gradient[idx];
	}
	static inline void backward(index<1> idx, table_view<> const& output, fixed_array<table_view<>, 1> const& inputs, state_t state) restrict(amp) {
		table_view<> input = inputs[0];

		float acc = 0.0f;
		float value = input.m_value[idx];

		// Matrix-vector multiplication
		for (int i = 0; i < state.weights.m_gradient.extent[0]; ++i) {
			index<2> idx2(i, idx[0]);

			float gradient = output.m_gradient[i];

			acc += state.weights.m_value[idx2] * gradient;

			// Update weight gradient
			state.weights.m_gradient[idx2] += gradient * value;
		}
		input.m_gradient[idx] = acc;
	}
	static inline void clearGradient(index<1> idx, table_view<> const& output, fixed_array<table_view<>, 1> const& inputs, state_t state) restrict(amp) {
		module_function::clearGradient(idx, output, inputs, module_function::state_t());
		state.bias.m_gradient[idx] = 0.0f;
		for (int i = 0; i < state.weights.m_gradient.extent[1]; ++i) {
			index<2> idx2(idx[0], i);
			state.weights.m_gradient[idx2] = 0.0f;
		}
	}

	inline state_t getState() {
		return state_t{ m_weights.view(), m_bias.view() };
	}
};

class module_input : public module {
protected:
	tensor<> m_output;
public:
	inline module_input(network* nn, extent<1> extent) : module(nn), m_output(nn, tensor_type_transient, extent) { }

	virtual void updateOutput() {
	}
	virtual void updateGradInput() {
	}

	void setValue(array_view<float, 1> value) {
		if (value.extent != m_output.extent())
			throw "Extent mismatch";
		value.copy_to(m_output.view().m_value);
	}
	inline tensor_view<> getOutput() {
		return tensor_view<>(m_output);
	}
};

template<int N, typename S>
class module_container : public module {
protected:
	fixed_array<table_view<>, N> m_outputs;
public:
	template<typename... P>
	inline module_container(network* nn, P&&... p) : module(nn), m_outputs(S::build(nn, std::forward<P>(p)...)) { }
	virtual void updateOutput() {
	}
	virtual void updateGradInput() {
	}
};

class module_lstm : public module_container<1, module_lstm> {
public:
	using module_container::module_container;

	static fixed_array<table_view<>, 1> build(network* nn) {

	}
};
