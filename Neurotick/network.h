#pragma once

class linear_tensor_allocator {
private:
	std::vector<tensor_base*> m_tensors;
	std::unique_ptr<table<1>> m_table;

public:
	void registerTensor(tensor_base* tensor) {
		m_tensors.push_back(tensor);
	}
	void allocate() {
		m_table = nullptr;

		unsigned totalSize = 0;
		for (auto tensor : m_tensors) {
			totalSize += tensor->size();
		}

		if (totalSize > 0) {
			m_table = std::make_unique<table<1>>(extent<1>(totalSize));
			table_view<1> tableView(*m_table);

			unsigned offset = 0;
			for (auto tensor : m_tensors) {
				unsigned size = tensor->size();
				tensor->setSource(tableView.section(index<1>(offset), extent<1>(size)));
				offset += size;
			}
		}
	}
};

class network {
private:
	std::vector<std::unique_ptr<module>> m_moduleSeq;
	bool m_isLearning;
	linear_tensor_allocator m_tensorAllocators[tensor_type_count];

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
	void updateOutput() {
		for (auto& module : m_moduleSeq)
			module->updateOutput();
	}
	void updateGradInput() {
		for (auto& module : boost::adaptors::reverse(m_moduleSeq))
			module->updateGradInput();
	}
	void compile() {
		for (int i = 0; i < tensor_type_count; ++i)
			m_tensorAllocators[i].allocate();
	}
	inline void registerTensor(tensor_type type, tensor_base* tensor) {
		m_tensorAllocators[type].registerTensor(tensor);
	}
};
