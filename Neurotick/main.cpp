// AmpTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "neurotick.h"

struct rms_config {
	float learningRate = 1e-2f;
	float alpha = 0.99f;
	float epsilon = 1e-4f;
};

float lerp(float alpha, float a, float b) restrict(amp) {
	return alpha*b + (1.0f - alpha)*a;
}

class rms_prop {
private:
	rms_config m_config;
	std::function<float ()> m_evaluate;
	table_view<> m_weights;
	array<float, 1> m_state;

public:
	rms_prop(std::function<float()> evaluate, table_view<> weights, rms_config const& config)
		: m_weights(weights), m_state(weights.extent()), m_config(config), m_evaluate(evaluate) {
		fill(array_view<float, 1>(m_state), 0.0f);
	}
	float step() {
		float loss = m_evaluate();
		auto config = m_config;
		auto state = array_view<float, 1>(m_state);
		auto weights = m_weights;

		parallel_for_each(weights.extent(), [=](index<1> idx) restrict(amp) {
			float gradient = weights.m_gradient[idx];
			float newState = lerp(config.alpha, state[idx], gradient*gradient);
			float rms = fast_math::sqrtf(newState) + config.epsilon;

			state[idx] = newState;
			weights.m_value[idx] -= config.learningRate * (gradient / rms);
		});

		clamp(weights.m_value, -5.0f, 5.0f);

		return loss;
	}
};

class char_vocab {
protected:
	int m_charToClass[256];
	int m_classToChar[256];
	int m_classCount;
public:
	char_vocab() {
		m_classCount = 0;
		for (int i = 0; i < 256; ++i) {
			m_charToClass[i] = -1;
			m_classToChar[i] = -1;
		}
	}
	char_vocab(std::string const& filename) {
		m_classCount = 0;
		for (int i = 0; i < 256; ++i) {
			m_charToClass[i] = -1;
			m_classToChar[i] = -1;
		}

		std::ifstream is(filename);
		int c;
		while ((c = is.get()) != std::char_traits<char>::eof()) {
			addChar(c);
		}
	}
	inline int addChar(int c) {
		if (c == -1)
			return -1;
		int index = (unsigned char)(char)c;
		if (m_charToClass[index] == -1) {
			m_classToChar[m_classCount] = index;
			m_charToClass[index] = m_classCount++;
		}
		return m_charToClass[index];
	}
	inline int getClass(int c) const {
		if (c == -1)
			return -1;
		return m_charToClass[(unsigned char)(char)c];
	}
	inline int getChar(int c) const {
		if (c == -1)
			return -1;
		return m_classToChar[c];
	}
	inline int getClassCount() const {
		return m_classCount;
	}

	inline void save(std::string const& filename, bool binary = true) const {
		std::ofstream os(filename, std::ios::trunc|(binary ? std::ios::binary : 0));
		if (binary) {
			os.write((char const*)this, sizeof(char_vocab));
		} else {
			os << m_classCount << std::endl;
			for (int i = 0; i < 256; ++i) {
				os << m_charToClass[i] << " " << m_classToChar[i] << std::endl;
			}
		}
	}
	inline void load(std::string const& filename, bool binary = true) {
		std::ifstream is(filename, binary ? std::ios::binary : 0);
		if (binary) {
			is.read((char*)this, sizeof(char_vocab));
		} else {
			is >> m_classCount;
			for (int i = 0; i < 256; ++i) {
				is >> m_charToClass[i] >> m_classToChar[i];
			}
		}
	}
};

class char_sequence_loader {
private:
	std::ifstream m_is;
	char_vocab m_vocab;
	int m_sequenceLength;
	int m_batchSize;
	std::vector<char> m_buffer;
	std::vector<int> m_classes;

public:
	char_sequence_loader(std::string filename, boost::optional<char_vocab> vocab, int sequenceLength, int batchSize)
		: m_is(filename), m_sequenceLength(sequenceLength), m_batchSize(batchSize), m_buffer(sequenceLength*batchSize + 1), m_classes(sequenceLength*batchSize + 1) {
		if (vocab.is_initialized()) {
			m_vocab = *vocab;
		} else {
			m_vocab = char_vocab(filename);
		}
	}
	inline char_vocab const& getVocab() const {
		return m_vocab;
	}
	inline bool next() {
		m_is.read(m_buffer.data(), m_buffer.size());
		if (m_is.fail())
			return false;
		m_is.putback(m_buffer.back());

		for (unsigned i = 0; i < m_buffer.size(); ++i)
			m_classes[i] = m_vocab.getClass((unsigned char)m_buffer[i]);

		return true;
	}
	inline void getInputBatch(int index, array_view<int, 1> classes) {
		for (int i = 0; i < m_batchSize; ++i) {
			classes[i] = m_classes[i*m_sequenceLength + index];
		}
	}
	inline void getTargetBatch(int index, array_view<int, 1> classes) {
		for (int i = 0; i < m_batchSize; ++i) {
			classes[i] = m_classes[i*m_sequenceLength + index + 1];
		}
	}
	inline void reset() {
		m_is.clear();
		m_is.seekg(0);
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

std::string ws2s(const std::wstring& wstr) {
	typedef std::codecvt_utf8<wchar_t> convert_typeX;
	std::wstring_convert<convert_typeX, wchar_t> converterX;

	return converterX.to_bytes(wstr);
}

template<typename T> void saveTable(std::string const& filename, array_view<T, 1> data, bool binary = true) {
	int openMode = std::ios::trunc;
	if (binary)
		openMode |= std::ios::binary;
	std::ofstream os(filename, openMode);
	if (binary) {
		os.write((char*)data.data(), sizeof(T)*data.extent.size());
	} else {
		for (unsigned i = 0; i < data.extent.size(); ++i) {
			os << data[i] << std::endl;
		}
	}
}

template<typename T> void loadTable(std::string const& filename, array_view<T, 1> data, bool binary = true) {
	std::ifstream is(filename, binary ? std::ios::binary : 0);
	if (binary) {
		is.read((char*)data.data(), sizeof(T)*data.extent.size());
	} else {
		for (unsigned i = 0; i < data.extent.size(); ++i) {
			is >> data[i];
		}
	}
}

int main(int argc, char* argv[])
{
	accelerator defaultDevice;
	std::cout << "Using device: " << ws2s(defaultDevice.get_description()) << std::endl;

	network nn;
	const bool learning = true;

	nn.setIsLearning(learning);

	const int sequenceLength = learning ? 16 : 1;
	const int batchSize = learning ? 16 : 1;
	const int rnnSize = 256;

	char_vocab vocab;
	if (learning) {
		vocab = char_vocab("data/input.txt");
		vocab.save("data/vocab.dat", false);
	} else {
		vocab.load("data/vocab.dat", false);
	}


	const int classCount = vocab.getClassCount();

	// Construct rnn
	auto embedding = nn.make<module_class_embedding>(classCount, extent<2>(batchSize, rnnSize));
	auto lstm = nn.make<module_lstm>(extent<1>(rnnSize), embedding);
	auto softmax = nn.make<module_softmax>(extent<1>(classCount), lstm);
	auto criterion = nn.make<module_class_nll_criterion>(softmax);

	sequence_evaluator evaluator(&nn, sequenceLength, criterion);

	if (learning) {
		char_sequence_loader loader("data/input.txt", boost::none, sequenceLength, batchSize);

		// Setup training scenario

		rms_config config;
		rms_prop prop([&]() {
			for (int i = 0; i < sequenceLength; ++i) {
				auto stateProvider = evaluator[i];
				loader.getInputBatch(i, embedding->getInput(stateProvider));
				loader.getTargetBatch(i, criterion->getTarget(stateProvider));

				//for (int j = 0; j < batchSize; ++j) {
				//	int c = embedding->getInput(stateProvider)[j];
				//	char ch = loader.getVocab().getChar(c);
				//	std::cout << ((ch == 10 || ch == 13) ? (char)' ' : ch);
				//}
				//std::cout << (char)' ';
				//for (int j = 0; j < batchSize; ++j) {
				//	int c = criterion->getTarget(stateProvider)[j];
				//	char ch = loader.getVocab().getChar(c);
				//	std::cout << ((ch == 10 || ch == 13) ? (char)' ' : ch);
				//}
				//std::cout << std::endl;
			}

			return evaluator.evaluate();
		}, *evaluator.getTensorView(0, tensor_type_weight), config);

		// Initialise weights
		cpuFill(evaluator.getTensorView(0, tensor_type_weight)->m_value, uniformRandom(-0.08f, 0.08f));

		// Run
		const int numEpochs = 1;

		int iteration = 0;
		for (int i = 0; i < numEpochs; ++i) {
			while (loader.next()) {
				float loss = prop.step();
				std::cout << "Iteration: " << iteration << ", Loss: " << loss << std::endl;

				if (iteration % 100 == 0) {
					saveTable("data/weights.dat", evaluator.getTensorView(0, tensor_type_weight)->m_value, false);
				}

				++iteration;
			}
			loader.reset();
		}
	} else {
		loadTable("data/weights.dat", evaluator.getTensorView(0, tensor_type_weight)->m_value, false);
		fill(evaluator.getTensorView(0, tensor_type_state)->m_value, 0.0f);

		auto stateProvider = evaluator[0];

		std::string primer = "Hello, my name is ";
		for (unsigned i = 0; i < primer.length(); ++i) {
			embedding->getInput(stateProvider)[0] = vocab.getClass((unsigned char)primer[i]);
			evaluator.evaluate();
		}

		char prevChar = ' ';
		while (true) {
			embedding->getInput(stateProvider)[0] = vocab.getClass((unsigned char)prevChar);
			evaluator.evaluate();
			auto outputs = softmax->getOutput().view(stateProvider);
			int bestClass = -1;
			float bestValue = -std::numeric_limits<float>::infinity();
			for (int i = 0; i < outputs.extent()[1]; ++i) {
				index<2> idx(0, i);
				float value = outputs.m_value[idx];
				if (value > bestValue) {
					bestValue = value;
					bestClass = i;
				}
			}

			prevChar = vocab.getChar(bestClass);
			std::cout << prevChar;
		}
	}

	getchar();
	return 0;
}
