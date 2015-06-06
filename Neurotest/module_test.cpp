#include "stdafx.h"
#include "CppUnitTest.h"
#include "../Neurotick/neurotick.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

class test_evaluator : public evaluator_base {
protected:
	tensor_view<2> m_output;
	std::vector<tensor_view<2>> m_inputs;

public:
	inline test_evaluator(network* nn, unsigned sequenceLength, tensor_view<2> output, std::vector<tensor_view<2>> inputs) : evaluator_base(nn, sequenceLength), m_output(output), m_inputs(inputs) {
	}
	void evaluate(std::vector<float> expected, const float epsilon = 1e-6f) {
		auto weights = getTensorView(0, tensor_type_weight);
		float loss = 0.0f;

		if (weights.is_initialized() && m_nn->getIsLearning())
			fill(weights->m_gradient, 0.0f);

		for (int i = 0; i < (int)m_sequenceLength; ++i) {
			auto stateProvider = (*this)[i];
			m_nn->updateOutput(stateProvider);
		}

		auto firstState = (*this)[0];
		auto lastState = (*this)[m_sequenceLength - 1];
		auto result = m_output.view(lastState);

		// Check output matches expected
		Assert::AreEqual(result.extent().size(), expected.size());

		float actualY = 0.0f;
		for (unsigned i = 0; i < expected.size(); ++i) {
			actualY += result.m_value[0][i];
			float delta = abs(result.m_value[0][i] - expected[i]);
			Assert::IsFalse(delta > epsilon);
		}

		if (m_nn->getIsLearning()) {
			fill(result.m_gradient, 1.0f);

			for (int i = (int)m_sequenceLength - 1; i >= 0; --i) {
				auto stateProvider = (*this)[i];
				m_nn->updateGradInput(stateProvider);
			}

			// Check gradients numerically
			std::vector<float> actualGradients;
			for (unsigned i = 0; i < m_inputs.size(); ++i) {
				auto input = m_inputs[i].view(firstState);
				for (unsigned j = 0; j < input.extent().size(); ++j) {
					actualGradients.push_back(input.m_gradient[0][j]);
				}
			}
			std::reverse(actualGradients.begin(), actualGradients.end());
			for (unsigned i = 0; i < m_inputs.size(); ++i) {
				auto input = m_inputs[i].view(firstState);
				for (unsigned j = 0; j < input.extent().size(); ++j) {
					const float dx = 1e-2f;
					float originalInput = input.m_value[0][j];
					input.m_value[0][j] = originalInput + dx;

					for (int k = 0; k < (int)m_sequenceLength; ++k) {
						auto stateProvider = (*this)[k];
						m_nn->updateOutput(stateProvider);
					}

					input.m_value[0][j] = originalInput;

					float newY = 0.0f;
					for (unsigned k = 0; k < result.extent().size(); ++k) {
						newY += result.m_value[0][k];
					}
					float gradient = (newY - actualY) / dx;
					float delta = abs(gradient - actualGradients.back());
					actualGradients.pop_back();

					Assert::IsFalse(delta > 1e-2);
				}
			}
		}
	}
};

namespace Neurotest
{		
	TEST_CLASS(ModuleTests)
	{
	public:
		
		TEST_METHOD(TestAdd)
		{
			network nn;
			auto a = nn.make<module_input>(extent<2>(1, 4));
			auto b = nn.make<module_input>(extent<2>(1, 4));
			auto result = nn.make<module_add<>>(a, b);
			
			test_evaluator evaluator(&nn, 1, result, { a, b });

			a->setValue(evaluator[0], { { 1, 2, 3, 4 } });
			b->setValue(evaluator[0], { { 2, 4, 8, 16 } });

			evaluator.evaluate({ 3, 6, 11, 20 });
		}

		TEST_METHOD(TestLogSoftMax)
		{
			network nn;
			auto a = nn.make<module_input>(extent<2>(1, 4));
			auto result = nn.make<module_log_soft_max>(a);

			test_evaluator evaluator(&nn, 1, result, { a });

			a->setValue(evaluator[0], { { 1, 2, 3, 4 } });

			const float e = 2.7182818284590f;
			float logSum = 1.0f + log(1.0f + e + e*e + e*e*e);
			evaluator.evaluate({ 1 - logSum, 2 - logSum, 3 - logSum, 4 - logSum });
		}

	};
}