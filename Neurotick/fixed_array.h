#pragma once

template<typename T, int N> struct fixed_array;

#define ENUM_ARRAY_INDEX_M(z, n, data) data[n]
#define ENUM_ARRAY_INDEX(count, data) BOOST_PP_ENUM(count, ENUM_ARRAY_INDEX_M, data)
#define ENUM_FIELD_INIT_M(z, n, data) m_ ## data ## n (data ## n)
#define ENUM_FIELD_INIT(count, data) BOOST_PP_ENUM(count, ENUM_FIELD_INIT_M, data)
#define ENUM_FIELD_DECL_M(z, n, data) data ## n;
#define ENUM_FIELD_DECL(count, data) BOOST_PP_REPEAT(count, ENUM_FIELD_DECL_M, data)
#define ENUM_CASE_M(z, n, data) case n: return data ## n;
#define ENUM_CASE(count, data) BOOST_PP_REPEAT(count, ENUM_CASE_M, data)
#define ENUM_APPLY_M(z, n, data) f(data ## n)
#define ENUM_APPLY(count, data) BOOST_PP_ENUM(count, ENUM_APPLY_M, data)

#define DEF_SPEC_ARRAY(z, n, data) \
	template<typename T> struct fixed_array<T, n> { \
	public: \
		ENUM_FIELD_DECL(n, T m_elem); \
		\
		inline fixed_array(BOOST_PP_ENUM_PARAMS(n, T const& elem)) restrict(amp, cpu) : ENUM_FIELD_INIT(n, elem) {}; \
		inline T const& operator[](int i) const restrict(amp, cpu) { \
			switch (i) { \
			ENUM_CASE(n, m_elem) \
			default: return m_elem0; \
			} \
		} \
	};

BOOST_PP_REPEAT(16, DEF_SPEC_ARRAY, ());

namespace detail {
	template<unsigned N> struct marker {};
	template<unsigned M, typename T, unsigned N, typename F, typename... P> auto fixedMapImpl(std::array<T, N> const& arr, F&& f, marker<M> unused, P const&... p) {
		return fixedMapImpl(arr, std::forward<F>(f), marker<M + 1>(), p..., std::get<M>(arr));
	}
	template<typename T, unsigned N, typename F, typename... P> auto fixedMapImpl(std::array<T, N> const& arr, F&& f, marker<N> unused, P const&... p) {
		typedef decltype(f(std::declval<T>())) U;
		return fixed_array<U, N>(f(p)...);
	}
}

template<typename T, unsigned N, typename F>
auto fixedMap(std::array<T, N> const& arr, F&& f) {
	return detail::fixedMapImpl(arr, std::forward<F>(f), detail::marker<0>());
}
