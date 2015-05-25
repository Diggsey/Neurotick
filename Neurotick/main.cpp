// AmpTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

class network;

#include "fixed_array.h"
#include "range_loop.h"
#include "tensor.h"
#include "module.h"
#include "network.h"

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

int _tmain(int argc, _TCHAR* argv[])
{
	network nn;

	extent<1> size(8);

	auto a = nn.make<module_input>(size);
	auto b = nn.make<module_add<2>>(a->getOutput(), a->getOutput());
	auto c = nn.make<module_div>(b->getOutput(), a->getOutput());

	nn.compile();

	array<float, 1> data(size, boost::make_counting_iterator(1.0f));

	a->setValue(data);
	nn.updateOutput();
	nn.updateGradInput();

	printArray(a->getOutput().view().m_value);
	printArray(b->getOutput().view().m_value);
	printArray(c->getOutput().view().m_value);

	getchar();
	return 0;
}
