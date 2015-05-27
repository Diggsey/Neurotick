// AmpTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "neurotick.h"

struct rms_config {
	float learningRate = 1e-2f;
	float alpha = 0.99f;
	float epsilon = 1e-8f;
};

float lerp(float alpha, float a, float b) restrict(amp) {
	return alpha*b + (1.0f - alpha)*a;
}

template<typename F>
class rms_prop {
private:
	rms_config m_config;
	F m_f;
	array_view<float, 1>& m_x;
	array<float, 1> m_state;
	array<float, 1> m_loss;

public:
	rms_prop(F f, array_view<float, 1>& x, rms_config const& config) : m_f(f), m_x(x), m_state(x.get_extent()), m_loss(x.get_extent()), m_config(config) {
		fill(m_state, 0.0f);
		fill(m_loss, 0.0f);
	}
	void step() {
		parallel_for_each(x.extent, [&](index<Rank> idx) restrict(amp) {
			m_state[idx] = lerp(m_config.alpha, m_state[idx], m_state[idx]);
		});
	}
};

template<int N, int M> extent<N + M> operator*(extent<N> const& a, extent<M> const& b) {
	extent<N + M> result;
	for (int i = 0; i < N; ++i) {
		result[i] = a[i];
	}
	for (int i = 0; i < M; ++i) {
		result[i + N] = b[i];
	}
	return result;
}

int main(int argc, char* argv[])
{
	network nn;

	extent<1> batchSize(16);
	extent<1> size(8);

	auto a = nn.make<module_input>(batchSize*size);
	auto b = nn.make<module_add<2>>(make_array( a->getOutput(), a->getOutput() ));
	auto c = nn.make<module_div>(b->getOutput(), a->getOutput());
	auto d = nn.make<module_rcp>(c->getOutput());
	auto e = nn.make<module_linear>(size, d->getOutput());

	batch_evaluator evaluator(&nn);

	/*
	auto weights = nn.getTensorView(tensor_type_weight);
	cpuFill(weights.m_value, uniformRandom(-0.08f, 0.08f));
	*/
	array<float, 2> data(batchSize*size, boost::make_counting_iterator(1.0f));

	a->setValue(evaluator[0], data);
	nn.updateOutput(evaluator[0]);
	nn.updateGradInput(evaluator[0]);

	printArray(a->getOutput().view(evaluator[0]).m_value);
	printArray(b->getOutput().view(evaluator[0]).m_value);
	printArray(c->getOutput().view(evaluator[0]).m_value);
	printArray(d->getOutput().view(evaluator[0]).m_value);
	printArray(e->getOutput().view(evaluator[0]).m_value);

	getchar();
	return 0;
}
