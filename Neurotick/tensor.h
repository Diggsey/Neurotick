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
template<typename T, typename U, typename F>
U reduce(array_view<T, 1> arr, U acc, F&& f) {
	for (int i = 0; i < arr.extent[0]; ++i) {
		acc = f(acc, arr[i]);
	}
	return acc;
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

class state_provider {
public:
	virtual table_view<> getTensorView(tensor_type type, index<1> offset, extent<1> size) const = 0;
	virtual table_view<> getNextState(index<1> offset, extent<1> size) const = 0;
};

template<int N = 1> class tensor {
private:
	tensor_type m_type;
	extent<N> m_extent;
	index<1> m_offset;

public:
	tensor(network* nn, tensor_type type, extent<N> extent) : m_type(type), m_extent(extent), m_offset(nn->allocateTensor(type, extent.size())) {
	}
	inline unsigned size() {
		return m_extent.size();
	}
	inline extent<N> extent() {
		return m_extent;
	}
	inline table_view<N> view(state_provider const& stateProvider) {
		return stateProvider.getTensorView(m_type, m_offset, concurrency::extent<1>(size())).view_as(m_extent);
	}
	inline table_view<N> nextView(state_provider const& stateProvider) {
		if (m_type == tensor_type_state)
			return stateProvider.getNextState(m_offset, concurrency::extent<1>(size())).view_as(m_extent);
		else
			throw "Invalid tensor type";
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
	inline table_view<N> view(state_provider const& stateProvider) const {
		return m_tensor->view(stateProvider);
	}
	inline table_view<N> nextView(state_provider const& stateProvider) const {
		return m_tensor->nextView(stateProvider);
	}
};
