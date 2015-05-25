#pragma once

template<unsigned N> class range_to;
const int uniform_param = 0x80000000;

template<unsigned N, int I> class param_indexer {
public:
	template<typename T> static auto& index(T& param) restrict(amp, cpu) {
		return param[N + I];
	}
};
template<unsigned N> class param_indexer<N, uniform_param> {
public:
	template<typename T> static T& index(T& param) restrict(amp, cpu) {
		return param;
	}
};
template<typename F>
class range_invoker {
public:
	F f;
	template<int... I, unsigned N, typename... P> inline void each(range_to<N> unused, P&... p) restrict(amp, cpu) {
		each<I...>(range_to<N - 1>(), p...);
		f(N - 1, param_indexer<N - 1, I>::index(p)...);
	}
	template<int... I, unsigned N, typename... P> inline void eachRev(range_to<N> unused, P&... p) restrict(amp, cpu) {
		f(N - 1, param_indexer<N - 1, I>::index(p)...);
		eachRev<I...>(range_to<N - 1>(), p...);
	}
	template<int... I, typename... P> inline void each(range_to<0> unused, P&... p) restrict(amp, cpu) { }
	template<int... I, typename... P> inline void eachRev(range_to<0> unused, P&... p) restrict(amp, cpu) { }
};
template<unsigned N> class range_to {
public:
	template<int... I, typename F, typename... P> static void each(F f, P&... p) restrict(amp, cpu) {
		range_invoker<F> invoker{ f };
		invoker.each<I...>(range_to<N>(), p...);
	}
	template<int... I, typename F, typename... P> static void eachRev(F f, P&... p) restrict(amp, cpu) {
		range_invoker<F> invoker{ f };
		invoker.eachRev<I...>(range_to<N>(), p...);
	}
};
