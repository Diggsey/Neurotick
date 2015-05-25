#pragma once

template<typename T, int Rank>
void fill(array_view<T, Rank> arr, T initValue) {
	parallel_for_each(arr.extent, [=](index<Rank> idx) restrict(amp) {
		arr[idx] = initValue;
	});
}
template<typename T, int Rank, typename F>
void fill(array_view<T, Rank> arr, F&& f) {
	parallel_for_each(arr.extent, std::forward<F>(f));
}
template<typename T, int Rank, typename F>
void cpuFill(array_view<T, Rank> arr, F&& f) {
	unsigned size = arr.extent.size();
	for (unsigned i = 0; i < size; ++i) {
		index<Rank> idx;
		unsigned j = i;
		for (int k = Rank-1; k >= 0; --k) {
			unsigned e = arr.extent[k];
			idx[k] = j % e;
			j = j / e;
		}
		arr[idx] = f(idx);
	}
	arr.synchronize();
}
auto uniformRandom(float minv, float maxv) {
	std::default_random_engine generator;
	std::uniform_real_distribution<float> dist(minv, maxv);
	return [=](auto idx) mutable {
		return dist(generator);
	};
}

void printArray(array_view<float, 1> view) {
	view.refresh();
	array_view<float, 1> temp(view.extent);
	view.copy_to(temp);
	for (int i = 0; i < temp.extent[0]; ++i)
		std::cout << std::setw(9) << std::setprecision(3) << temp[i];
	std::cout << std::endl;
}

template<int N = 1>
struct table {
	array<float, N> m_value;
	array<float, N> m_gradient;

	inline table(extent<N> size) : m_value(size), m_gradient(size) { }
	inline extent<N> extent() const { return m_value.extent; }
};

template<int N = 1>
struct table_view {
public:
	array_view<float, N> m_value;
	array_view<float, N> m_gradient;

	inline table_view(table<N>& src) : m_value(src.m_value), m_gradient(src.m_gradient) { }
	inline table_view(array_view<float, N> value, array_view<float, N> gradient) : m_value(value), m_gradient(gradient) { }
	inline extent<N> extent() const { return m_value.extent; }
	template<int M>
	table_view<M> view_as(concurrency::extent<M> extent) {
		return table_view<M>(m_value.view_as(extent), m_gradient.view_as(extent));
	}
	table_view<N> section(concurrency::index<N> index, concurrency::extent<N> extent) {
		return table_view<N>(m_value.section(index, extent), m_gradient.section(index, extent));
	}
};


enum tensor_type {
	tensor_type_transient = 0,
	tensor_type_state,
	tensor_type_weight,
	tensor_type_count
};

class tensor_base {
public:
	virtual unsigned size() = 0;
	virtual void setSource(table_view<1> source) = 0;
};
template<int N = 1> class tensor : public tensor_base {
private:
	boost::optional<table_view<N>> m_data;
	extent<N> m_extent;

public:
	tensor(network* nn, tensor_type type, extent<N> extent) : m_data(), m_extent(extent) {
		nn->registerTensor(type, this);
	}
	virtual unsigned size() {
		return m_extent.size();
	}
	virtual void setSource(table_view<1> source) {
		m_data = source.view_as(m_extent);
	}
	inline extent<N> extent() {
		return m_extent;
	}
	inline table_view<N> view() {
		return *m_data;
	}
};
template<int N = 1> class tensor_view {
private:
	tensor<N>* m_tensor;

public:
	tensor_view(tensor<N>& tensor) : m_tensor(&tensor) { }

	inline extent<N> extent() const {
		return m_tensor->extent();
	}
	inline table_view<N> view() const {
		return m_tensor->view();
	}
};
