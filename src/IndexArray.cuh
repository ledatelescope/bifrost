/*
 * Copyright (c) 2016, The Bifrost Authors. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of The Bifrost Authors nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

template<int N, int M>
struct static_min {
	static constexpr int value = (N < M ? N : M);
};
template<int N, int M>
struct static_max {
	static constexpr int value = (N < M ? M : N);
};

template<typename I, I... VALS>
struct StaticIndexArray {
	typedef I type;
	static constexpr int size = sizeof...(VALS);
	//const I values[sizeof...(VALS...)] = {VALS...};
	static const I values[sizeof...(VALS)];
	//inline static const I at(int i) { return values[i]; }
/*
private:
	static constexpr I _values[] = {VALS...};
public:
	inline static constexpr I at(int i) { return _values[i]; }
*/
};
//template<typename I, I... Vals>
//static constexpr I StaticIndexArray::values[] = {Vals...};
template<typename I, I... VALS>
const I StaticIndexArray<I,VALS...>::values[sizeof...(VALS)] = {VALS...};

// TODO: Can only align this when N is a power of 2
template<typename T, int N>
class IndexArray {
	T vals_[N];
	typedef IndexArray self_type;
public:
	typedef T                 value_type;
	typedef value_type&       reference;
	typedef value_type const& const_reference;
	enum { SIZE = N };
	inline __host__ __device__
	IndexArray() {}
	
	template<typename Y>
	inline __host__ __device__
	explicit IndexArray(Y val) {
		for( int i=0; i<N; ++i ) {
			vals_[i] = val;
		}
	}
	
	template<typename Y>
	inline __host__ __device__
	IndexArray(Y vals[N]) {
		for( int i=0; i<N; ++i ) {
			vals_[i] = vals[i];
		}
	}
	//template<typename... Inds>
	//inline __host__ __device__
	//IndexArray(Inds... inds) : vals_({inds...}) {}
	inline __host__ __device__
	const_reference operator[](int i) const { return vals_[i]; }
	inline __host__ __device__
	reference       operator[](int i)       { return vals_[i]; }
	
	template<int M>
	IndexArray<T,N+M> join(IndexArray<T,M> const& other) {
		T vals[N+M];
		for( int i=0; i<N; ++i ) {
			vals[i] = vals_[i];
		}
		for( int j=0; j<M; ++j ) {
			vals[N+j] = other[j];
		}
		return IndexArray<T,N+M>(vals);
	}
	
	inline __host__ __device__
	self_type& operator-() {
		for( int i=0; i<N; ++i ) {
			vals_[i] = -vals_[i];
		}
		return *this;
	}
#define DEFINE_BINARY_UPDATE_OPERATOR(op) \
	template<typename Y> \
	inline __host__ __device__ \
	self_type& operator op(IndexArray<Y,N> const& other) { \
		for( int i=0; i<N; ++i ) { \
			vals_[i] op other[i]; \
		} \
		return *this; \
	}
	DEFINE_BINARY_UPDATE_OPERATOR(+=)
	DEFINE_BINARY_UPDATE_OPERATOR(-=)
	DEFINE_BINARY_UPDATE_OPERATOR(*=)
	DEFINE_BINARY_UPDATE_OPERATOR(/=)
	DEFINE_BINARY_UPDATE_OPERATOR(%=)
	DEFINE_BINARY_UPDATE_OPERATOR(|=)
	DEFINE_BINARY_UPDATE_OPERATOR(&=)
	DEFINE_BINARY_UPDATE_OPERATOR(^=)
	DEFINE_BINARY_UPDATE_OPERATOR(<<=)
	DEFINE_BINARY_UPDATE_OPERATOR(>>=)
#undef DEFINE_BINARY_UPDATE_OPERATOR
};

#define DEFINE_BINARY_OPERATOR(op) \
template<typename T, int N> \
inline __host__ __device__ \
IndexArray<T,N> operator op(IndexArray<T,N> a, IndexArray<T,N> b) { \
	IndexArray<T,N> c = a; \
	c op##= b; \
	return c; \
} \
template<typename T, int N> \
inline __host__ __device__ \
IndexArray<T,N> operator op(IndexArray<T,N> a, T b) { \
	IndexArray<T,N> c = a; \
	c op##= IndexArray<T,N>(b); \
	return c; \
} \
template<typename T, int N> \
inline __host__ __device__ \
IndexArray<T,N> operator op(T a, IndexArray<T,N> b) { \
	IndexArray<T,N> c = IndexArray<T,N>(a); \
	c op##= b; \
	return c; \
}
DEFINE_BINARY_OPERATOR(+)
DEFINE_BINARY_OPERATOR(-)
DEFINE_BINARY_OPERATOR(*)
DEFINE_BINARY_OPERATOR(/)
DEFINE_BINARY_OPERATOR(%)
DEFINE_BINARY_OPERATOR(|)
DEFINE_BINARY_OPERATOR(&)
DEFINE_BINARY_OPERATOR(^)
DEFINE_BINARY_OPERATOR(<<)
DEFINE_BINARY_OPERATOR(>>)
#undef DEFINE_BINARY_OPERATOR
