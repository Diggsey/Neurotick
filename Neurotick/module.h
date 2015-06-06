#pragma once

class module {
protected:
	network* m_network;

	inline module(network* nn);

public:
	virtual inline void updateOutput(state_provider const& stateProvider) = 0;
	virtual inline void updateGradInput(state_provider const& stateProvider) = 0;

	inline network* getNetwork();
};

template<int N, typename S>
class module_function : public module {
protected:
	tensor<2> m_output;
	std::array<tensor_view<2>, N> m_inputs;

	inline module_function(network* nn, extent<2> extent, std::array<tensor_view<2>, N> inputs);

public:
	struct state_t {};

	inline module_function(network* nn, std::array<tensor_view<2>, N> inputs);
	inline module_function(network* nn, tensor_view<2> input);
	inline module_function(network* nn, tensor_view<2> input1, tensor_view<2> input2);

	virtual inline void updateOutput(state_provider const& stateProvider);
	virtual inline void updateGradInput(state_provider const& stateProvider);

	inline tensor_view<2> getOutput();
	inline state_t getState(state_provider const& stateProvider);
	static inline void clearGradient(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, N> const& inputs, state_t state) restrict(amp);
	static inline extent<2> extent(std::array<tensor_view<2>, N> inputs);
};

// Base class for modules of the form `foldl <op> [inputs...]`,
// which includes scalar arithmetic
template<int N, typename S>
class module_scalar : public module_function<N, S> {
public:
	using module_function::module_function;

	template<bool unused = false>
	static inline void forward(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, N> const& inputs, state_t state) restrict(amp);
	template<bool unused = false>
	static inline void backward(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, N> const& inputs, state_t state) restrict(amp);
};

template<typename S>
class module_unary : public module_scalar<1, S> {
public:
	inline module_unary(network* nn, tensor_view<2> input);

	static inline void forward(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp);
	static inline void backward(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp);
};

template<typename S>
class module_binary : public module_scalar<2, S> {
public:
	inline module_binary(network* nn, tensor_view<2> input1, tensor_view<2> input2);
};

template<int N = 2>
class module_add : public module_scalar<N, module_add<N>> {
public:
	using module_scalar::module_scalar;

	static inline float identity() restrict(amp);
	static inline float op(float a, float b) restrict(amp);
	static inline float dop0(float a, float b) restrict(amp);
	static inline float dop1(float a, float b) restrict(amp);
};

class module_sub : public module_binary<module_sub> {
public:
	using module_binary::module_binary;

	static inline float identity() restrict(amp);
	static inline float op(float a, float b) restrict(amp);
	static inline float dop0(float a, float b) restrict(amp);
	static inline float dop1(float a, float b) restrict(amp);
};
class module_neg : public module_unary<module_neg> {
public:
	using module_unary::module_unary;

	static inline float op(float input) restrict(amp);
	static inline float dop(float output, float input) restrict(amp);
};

template<int N = 2>
class module_mul : public module_scalar<N, module_mul<N>> {
public:
	using module_scalar::module_scalar;

	static inline float identity() restrict(amp);
	static inline float op(float a, float b) restrict(amp);
	static inline float dop0(float a, float b) restrict(amp);
	static inline float dop1(float a, float b) restrict(amp);
};

class module_div : public module_binary<module_div> {
public:
	using module_binary::module_binary;

	static inline float identity() restrict(amp);
	static inline float op(float a, float b) restrict(amp);
	static inline float dop0(float a, float b) restrict(amp);
	static inline float dop1(float a, float b) restrict(amp);
};
class module_rcp : public module_unary<module_rcp> {
public:
	using module_unary::module_unary;

	static inline float op(float input) restrict(amp);
	static inline float dop(float output, float input) restrict(amp);
};
class module_sigmoid : public module_unary<module_sigmoid> {
public:
	using module_unary::module_unary;

	static inline float op(float input) restrict(amp);
	static inline float dop(float output, float input) restrict(amp);
};
class module_tanh : public module_unary<module_tanh> {
public:
	using module_unary::module_unary;

	static inline float op(float input) restrict(amp);
	static inline float dop(float output, float input) restrict(amp);
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
	virtual inline void updateGradInput(state_provider const& stateProvider);

	inline module_linear(network* nn, concurrency::extent<1> extent, tensor_view<2> input);

	static inline void forward(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp);
	static inline void backwardBias(index<1> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp);
	static inline void backwardWeights(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp);
	static inline void backward(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp);
	static inline void clearGradient(index<2> idx, table_view<2> const& output, fixed_array<table_view<2>, 1> const& inputs, state_t state) restrict(amp);

	inline state_t getState(state_provider const& stateProvider);
};
class module_log_soft_max : public module {
protected:
	tensor<2> m_output;
	tensor<2> m_scratch;
	tensor<1> m_maxInput;
	tensor_view<2> m_input;
public:
	inline module_log_soft_max(network* nn, tensor_view<2> input);

	virtual inline void updateOutput(state_provider const& stateProvider);
	virtual inline void updateGradInput(state_provider const& stateProvider);

	inline tensor_view<2> getOutput();
};

class module_input : public module {
protected:
	tensor<2> m_output;
public:
	inline module_input(network* nn, extent<2> extent);

	virtual inline void updateOutput(state_provider const& stateProvider);
	virtual inline void updateGradInput(state_provider const& stateProvider);

	inline void setValue(state_provider const& stateProvider, array_view<float, 2> value);
	inline void setValue(state_provider const& stateProvider, std::vector<std::vector<float>> value);
	inline void getGradient(state_provider const& stateProvider, array_view<float, 2> gradient);
	inline tensor_view<2> getOutput();
};

class module_class_embedding : public module {
protected:
	tensor<2> m_output;
	tensor<2> m_weights;
	buffer<int> m_input;
public:
	inline module_class_embedding(network* nn, unsigned numClasses, extent<2> extent);

	virtual inline void updateOutput(state_provider const& stateProvider);
	virtual inline void updateGradInput(state_provider const& stateProvider);

	inline void setInput(state_provider const& stateProvider, array_view<int, 1> classes);
	inline array_view<int, 1> getInput(state_provider const& stateProvider);
	inline tensor_view<2> getOutput();
};

class module_state : public module {
	friend class module_state_input;
protected:
	tensor<2> m_output;
	bool m_hasInput;
public:
	inline module_state(network* nn, concurrency::extent<2> extent);

	virtual inline void updateOutput(state_provider const& stateProvider);
	virtual inline void updateGradInput(state_provider const& stateProvider);

	inline tensor_view<2> getOutput();
	inline void setInput(tensor_view<2> input);
};

class module_state_input : public module {
protected:
	tensor_view<2> m_state;
	tensor_view<2> m_input;
public:
	inline module_state_input(network* nn, module_state* state, tensor_view<2> input);

	virtual inline void updateOutput(state_provider const& stateProvider);
	virtual inline void updateGradInput(state_provider const& stateProvider);
};

template<int N, int M, typename S>
class module_container : public module {
protected:
	std::array<tensor_view<M>, N> m_outputs;
public:
	template<typename... P>
	inline module_container(network* nn, P&&... p);
	virtual inline void updateOutput(state_provider const& stateProvider);
	virtual inline void updateGradInput(state_provider const& stateProvider);
};

template<int M, typename S>
class module_container_function : public module_container<1, M, S> {
public:
	using module_container::module_container;

	inline tensor_view<M> getOutput();
};

class module_lstm : public module_container_function<2, module_lstm> {
public:
	using module_container_function::module_container_function;

	static inline std::array<tensor_view<2>, 1> build(network* nn, extent<1> extent, tensor_view<2> input);
};

class module_softmax : public module_container_function<2, module_softmax> {
public:
	using module_container_function::module_container_function;

	static inline std::array<tensor_view<2>, 1> build(network* nn, extent<1> extent, tensor_view<2> input);
};

class module_class_nll_criterion : public module {
protected:
	tensor_view<2> m_input;
	buffer<int> m_target;
	buffer<float> m_output;
public:
	inline module_class_nll_criterion(network* nn, tensor_view<2> input);

	virtual inline void updateOutput(state_provider const& stateProvider);
	virtual inline void updateGradInput(state_provider const& stateProvider);

	inline void setTarget(state_provider const& stateProvider, array_view<int, 1> classes);
	inline array_view<int, 1> getTarget(state_provider const& stateProvider);
	inline buffer_view<float> getOutput();
};
