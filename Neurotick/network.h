#pragma once

class storage_allocator {
private:
	unsigned m_size;

public:
	storage_allocator() : m_size(0) { }

	unsigned allocate(unsigned size) {
		auto result = m_size;
		m_size += size;
		return result;
	}
	inline unsigned size() const {
		return m_size;
	}
};

class network {
private:
	std::vector<std::unique_ptr<module>> m_moduleSeq;
	bool m_isLearning;
	storage_allocator m_tensorAllocators[tensor_type_count];
	storage_allocator m_bufferAllocators[buffer_type_count];

	template<typename T>
	inline T* addModule(std::unique_ptr<T>&& m) {
		T* result = m.get();
		m_moduleSeq.push_back(std::move(m));
		return result;
	}

public:
	inline network() : m_isLearning(true) {
	}
	inline bool getIsLearning() {
		return m_isLearning;
	}
	inline void setIsLearning(bool isLearning) {
		m_isLearning = isLearning;
	}

	template<typename T, typename... P> inline T* make(P&&... args) {
		return addModule(std::make_unique<T>(this, std::forward<P>(args)...));
	}
	void updateOutput(state_provider const& stateProvider) {
		for (auto& module : m_moduleSeq)
			module->updateOutput(stateProvider);
	}
	void updateGradInput(state_provider const& stateProvider) {
		for (auto& module : boost::adaptors::reverse(m_moduleSeq))
			module->updateGradInput(stateProvider);
	}
	inline unsigned allocateTensor(tensor_type type, unsigned size) {
		return m_tensorAllocators[type].allocate(size);
	}
	inline unsigned allocateBuffer(buffer_type type, unsigned size) {
		return m_bufferAllocators[type].allocate(size);
	}
	inline storage_allocator& getTensorAllocator(tensor_type type) {
		return m_tensorAllocators[type];
	}
	inline storage_allocator& getBufferAllocator(buffer_type type) {
		return m_bufferAllocators[type];
	}
};

class sequence_evaluator_state_provider : public state_provider {
	friend class sequence_evaluator;
protected:
	sequence_evaluator const* m_evaluator;
	unsigned m_idx;

	inline sequence_evaluator_state_provider(sequence_evaluator const* evaluator, unsigned idx) : m_evaluator(evaluator), m_idx(idx) { }
public:
	virtual table_view<> getTensorView(tensor_type type, index<1> offset, extent<1> size) const;
	virtual table_view<> getNextState(index<1> offset, extent<1> size) const;
	virtual array_view<int, 1> getBufferView(buffer_type type, index<1> offset, extent<1> size) const;
};

class sequence_evaluator {
protected:
	network* m_nn;
	unsigned m_sequenceLength;
	unsigned m_tupleSizes[tensor_type_count];
	std::unique_ptr<table<>> m_tuples[tensor_type_count];
	unsigned m_bufferSizes[buffer_type_count];
	std::unique_ptr<array<int, 1>> m_buffers[buffer_type_count];

public:
	inline void createTuple(tensor_type type, unsigned sequenceLength) {
		storage_allocator& alloc = m_nn->getTensorAllocator(type);
		m_tupleSizes[type] = alloc.size();
		if (m_tupleSizes[type] > 0) {
			m_tuples[type] = std::make_unique<table<>>(extent<1>(m_tupleSizes[type] * sequenceLength));
		}
	}
	inline void createBuffer(buffer_type type, unsigned sequenceLength) {
		storage_allocator& alloc = m_nn->getBufferAllocator(type);
		m_bufferSizes[type] = alloc.size();
		if (m_bufferSizes[type] > 0) {
			m_buffers[type] = std::make_unique<array<int, 1>>(extent<1>(m_bufferSizes[type] * sequenceLength));
		}
	}
	inline sequence_evaluator(network* nn, unsigned sequenceLength = 16) : m_nn(nn), m_sequenceLength(sequenceLength) {
		for (int i = 0; i < tensor_type_count; ++i)
			m_tupleSizes[i] = 0;

		createTuple(tensor_type_weight, 1);
		createTuple(tensor_type_state, sequenceLength + 1);
		createTuple(tensor_type_transient, sequenceLength);

		for (int i = 0; i < buffer_type_count; ++i) {
			m_bufferSizes[i] = 0;
			createBuffer((buffer_type)i, sequenceLength);
		}
	}
	inline boost::optional<table_view<>> getTensorView(unsigned idx, tensor_type type) const {
		switch (type) {
		case tensor_type_weight:
			idx = 0;
			break;
		}

		if (m_tuples[type]) {
			table_view<> view(*m_tuples[type]);
			unsigned size = m_tupleSizes[type];
			return view.section(index<1>(idx*size), extent<1>(size));
		} else {
			return boost::none;
		}
	}
	inline boost::optional<array_view<int, 1>> getBufferView(unsigned idx, buffer_type type) const {
		if (m_buffers[type]) {
			array_view<int, 1> view(*m_buffers[type]);
			unsigned size = m_bufferSizes[type];
			return view.section(index<1>(idx*size), extent<1>(size));
		} else {
			return boost::none;
		}
	}
	inline sequence_evaluator_state_provider operator[](unsigned idx) const {
		return sequence_evaluator_state_provider(this, idx);
	}
	table_view<> evaluate() {
		auto weights = getTensorView(0, tensor_type_weight);
		float loss = 0.0f;

		if (weights.is_initialized())
			fill(weights->m_gradient, 0.0f);
		
		for (int i = 0; i < (int)m_sequenceLength; ++i) {
			m_nn->updateOutput((*this)[i]);
		}
		for (int i = (int)m_sequenceLength - 1; i >= 0; --i) {
			m_nn->updateGradInput((*this)[i]);
		}
	}
};

table_view<> sequence_evaluator_state_provider::getTensorView(tensor_type type, index<1> offset, extent<1> size) const {
	return m_evaluator->getTensorView(m_idx, type)->section(offset, size);
}
table_view<> sequence_evaluator_state_provider::getNextState(index<1> offset, extent<1> size) const {
	return m_evaluator->getTensorView(m_idx + 1, tensor_type_state)->section(offset, size);
}
array_view<int, 1> sequence_evaluator_state_provider::getBufferView(buffer_type type, index<1> offset, extent<1> size) const {
	return m_evaluator->getBufferView(m_idx, type)->section(offset, size);
}
