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

	template<typename... T>
	inline module_function(network* nn, extent<1> extent, T... inputs) : module(nn), m_output(nn, tensor_type_transient, extent), m_inputs({ inputs... }) { }

public:
	template<typename... T>
	inline module_function(network* nn, T... inputs) : module(nn), m_output(nn, tensor_type_transient, S::extent({ inputs... })), m_inputs({ inputs... }) { }

	virtual void updateOutput() {
		auto inputs = fixedMap(m_inputs, [](auto& x) { return x.view(); });
		table_view<> output = m_output.view();


		try {
			// Update outputs
			output.m_value.discard_data();

			parallel_for_each(
				output.extent(),
				[=](index<1> idx) restrict(amp) {
				S::forward(idx, output, inputs);
			}
			);

			if (m_network->getIsLearning()) {
				// Clear gradients
				for (int i = 0; i < N; ++i)
					inputs[i].m_gradient.discard_data();

				parallel_for_each(
					output.extent(),
					[=](index<1> idx) restrict(amp) {
					range_to<N>::each<0>([=](int i, auto& inputi) restrict(amp) {
						inputi.m_gradient[idx] = 0.0f;
					}, inputs);
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

		try {
			parallel_for_each(
				output.extent(),
				[=](index<1> idx) restrict(amp) {
					S::backward(idx, output, inputs);
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
};

// Base class for modules of the form `foldl <op> [inputs...]`,
// which includes scalar arithmetic
template<int N, typename S>
class module_scalar : public module_function<N, S> {
public:
	using module_function::module_function;

	static inline extent<1> extent(std::initializer_list<tensor_view<>> inputs) {
		auto extent = inputs.begin()->extent();
		for (auto& input : inputs) {
			if (input.extent() != extent)
				throw "Extent mismatch";
		}
		return extent;
	}

	template<bool unused = false>
	static inline void forward(index<1> idx, table_view<> const& output, fixed_array<table_view<>, N> const& inputs) restrict(amp) {
		// Default implementation of `forward` is essentially `foldl <op> [inputs...]`

		float acc = S::identity();
		range_to<N>::each<uniform_param, 0>([=](int i, float& acc, auto& inputi) restrict(amp) {
			acc = S::op(acc, inputi.m_value[idx]);
		}, acc, inputs);
		output.m_value[idx] = acc;
	}
	template<bool unused = false>
	static inline void backward(index<1> idx, table_view<> const& output, fixed_array<table_view<>, N> const& inputs) restrict(amp) {
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
	using module_scalar::module_scalar;

	static inline void forward(index<1> idx, table_view<> const& output, fixed_array<table_view<>, 1> const& inputs) restrict(amp) {
		output.m_value[idx] = S::op(inputs[0].m_value[idx]);
	}
	static inline void backward(index<1> idx, table_view<> const& output, fixed_array<table_view<>, 1> const& inputs) restrict(amp) {
		inputs[0].m_gradient[idx] += output.m_gradient[idx] * S::dop(output.m_value[idx], inputs[0].m_value[idx]);
	}
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

class module_sub : public module_scalar<2, module_sub> {
public:
	using module_scalar::module_scalar;

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

class module_div : public module_scalar<2, module_div> {
public:
	using module_scalar::module_scalar;

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
public:
	template<typename... T>
	inline module_linear(network* nn, extent<1> extent, T... inputs) : module_function(nn, extent, inputs...) { }

	static inline void forward(index<1> idx, table_view<> const& output, fixed_array<table_view<>, 1> const& inputs) restrict(amp) {
		// TODO
	}
	static inline void backward(index<1> idx, table_view<> const& output, fixed_array<table_view<>, 1> const& inputs) restrict(amp) {
		// TODO
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
