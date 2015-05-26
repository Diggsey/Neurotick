#pragma once

class tensor_allocator {
private:
	unsigned m_size;

public:
	tensor_allocator() : m_size(0) { }

	unsigned allocateTensor(unsigned size) {
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
	tensor_allocator m_tensorAllocators[tensor_type_count];

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
		return m_tensorAllocators[type].allocateTensor(size);
	}
	inline tensor_allocator& getTensorAllocator(tensor_type type) {
		return m_tensorAllocators[type];
	}
};

class batch_evaluator_state_provider : public state_provider {
	friend class batch_evaluator;
protected:
	batch_evaluator const* m_evaluator;
	unsigned m_idx;

	inline batch_evaluator_state_provider(batch_evaluator const* evaluator, unsigned idx) : m_evaluator(evaluator), m_idx(idx) { }
public:
	virtual table_view<> getTensorView(tensor_type type, index<1> offset, extent<1> size) const;
	virtual table_view<> getNextState(index<1> offset, extent<1> size) const;
};

class batch_evaluator {
protected:
	network* m_nn;
	unsigned m_batchSize;
	unsigned m_sizes[tensor_type_count];
	std::unique_ptr<table<>> m_tables[tensor_type_count];

public:
	inline void createTable(tensor_type type, unsigned batchSize) {
		tensor_allocator& alloc = m_nn->getTensorAllocator(type);
		m_sizes[type] = alloc.size();
		if (m_sizes[type] > 0) {
			m_tables[type] = std::make_unique<table<>>(extent<1>(m_sizes[type] * batchSize));
		}
	}
	inline batch_evaluator(network* nn, unsigned batchSize = 16) : m_nn(nn), m_batchSize(batchSize) {
		for (int i = 0; i < tensor_type_count; ++i)
			m_sizes[i] = 0;

		createTable(tensor_type_weight, 1);
		createTable(tensor_type_state, batchSize + 1);
		createTable(tensor_type_transient, batchSize);
	}
	inline boost::optional<table_view<>> getView(unsigned idx, tensor_type type) const {
		switch (type) {
		case tensor_type_weight:
			idx = 0;
			break;
		}

		if (m_tables[type]) {
			table_view<> view(*m_tables[type]);
			unsigned size = m_sizes[type];
			return view.section(index<1>(idx*size), extent<1>(size));
		} else {
			return boost::none;
		}
	}
	inline batch_evaluator_state_provider operator[](unsigned idx) const {
		return batch_evaluator_state_provider(this, idx);
	}
	table_view<> evaluate() {
		auto weights = getView(0, tensor_type_weight);
		float loss = 0.0f;

		if (weights.is_initialized())
			fill(weights->m_gradient, 0.0f);
		
		for (int i = 0; i < (int)m_batchSize; ++i) {
			m_nn->updateOutput((*this)[i]);
		}
		for (int i = (int)m_batchSize-1; i >= 0; --i) {
			m_nn->updateGradInput((*this)[i]);
		}
	}
};

table_view<> batch_evaluator_state_provider::getTensorView(tensor_type type, index<1> offset, extent<1> size) const {
	return m_evaluator->getView(m_idx, type)->section(offset, size);
}
table_view<> batch_evaluator_state_provider::getNextState(index<1> offset, extent<1> size) const {
	return m_evaluator->getView(m_idx + 1, tensor_type_state)->section(offset, size);
}
