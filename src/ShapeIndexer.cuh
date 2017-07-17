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

#include "int_fastdiv.h"
#include "IndexArray.cuh"

namespace ShapeIndexer_detail {
// TODO: De-dupe this with the one in ArrayIndexer
template<bool B, class T = void>
struct enable_if {};
template<class T>
struct enable_if<true, T> { typedef T type; };

} // namespace ShapeIndexer_detail

template<typename T, int N, T... Vals>
struct GetPackIndex {};
template<typename T, int N, T First, T... Rest>
struct GetPackIndex<T,N,First,Rest...> {
	static constexpr T value = GetPackIndex<T,N-1,Rest...>::value;
};
template<typename T, T First, T... Rest>
struct GetPackIndex<T,0,First,Rest...> {
	static constexpr T value = First;
};

template<class Inds, int N>
struct GetIndex {};
template<typename T, T... Vals, int N>
struct GetIndex<StaticIndexArray<T,Vals...>,N> : GetPackIndex<T,N,Vals...> {};

//static_assert(GetIndex<StaticIndexArray<int,6,7,8>,1>::value == 7, "FAIL");

// Yay templates!
template<class >
struct PopFront {};
template<typename I, I Ind, I... Rest>
struct PopFront<StaticIndexArray<I,Ind,Rest...>> {
	using type = StaticIndexArray<I,Rest...>;
};
template<class >
struct GetFront {};
template<typename I, I Ind, I... Rest>
struct GetFront<StaticIndexArray<I,Ind,Rest...>> {
	static constexpr I value = Ind;
};
template<typename I>
struct GetFront<StaticIndexArray<I>> {
	static constexpr I value = 1;
};
template<typename I, class , I >
struct PushFront {};
template<typename I, I Ind, I... Inds>
struct PushFront<I,StaticIndexArray<I,Inds...>,Ind> {
	using type = StaticIndexArray<I,Ind,Inds...>;
};
template<class Src, class Dst=StaticIndexArray<typename Src::type>>
struct Reverse {};
template<typename I, I... Stem, I... Tail>
// Recursion pops from the front of Src and pushes to the front of Dst
struct Reverse<StaticIndexArray<I,Stem...>,StaticIndexArray<I,Tail...>>
     : Reverse<typename PopFront<StaticIndexArray<I,Stem...>>::type,
               typename PushFront<I,StaticIndexArray<I,Tail...>,
                                 GetFront<StaticIndexArray<I,Stem...>>::value>::type> {};
// Recursion ends when Src has no elements left
template<typename I, I... Tail>
struct Reverse<StaticIndexArray<I>,StaticIndexArray<I,Tail...>> {
	// Return Dst
	using type = StaticIndexArray<I,Tail...>;
};

template<class Src, class Dst=StaticIndexArray<typename Src::type,1>, class Sfinae=void>
struct StaticExclusiveScan {};
// Recursion pops a value off the front of Src, multiplies it with the front of
//   Dst, and pushes the result onto the front of Dst.
template<typename I, I... Stem, I... Tail>
struct StaticExclusiveScan<StaticIndexArray<I,Stem...>,StaticIndexArray<I,Tail...>,
                           typename ShapeIndexer_detail::enable_if<(sizeof...(Stem)>1)>::type>
     : StaticExclusiveScan<typename PopFront<StaticIndexArray<I,Stem...>>::type,
               typename PushFront<I,StaticIndexArray<I,Tail...>,
                                  GetFront<StaticIndexArray<I,Tail...>>::value *
                                  GetFront<StaticIndexArray<I,Stem...>>::value>::type> {};
// Recursion ends when Src has only 1 element left
template<typename I, I Stem, I... Tail>
struct StaticExclusiveScan<StaticIndexArray<I,Stem>,StaticIndexArray<I,Tail...>> {
	// Return Dst
	using type = StaticIndexArray<I,Tail...>;
};

template<typename T>
using StaticAccumulateShape = StaticExclusiveScan<typename Reverse<T>::type>;
/*
typedef StaticIndexArray<int,3,4,5> _Shape;
typedef Reverse<_Shape>::type _RevShape;
static_assert(_RevShape::values[0] == 5, "FAIL");
typedef StaticExclusiveScan<_RevShape>::type _CumuShape1;
static_assert(_CumuShape1::size == 3, "WRONG SHAPE");
static_assert(_CumuShape1::values[0] == 20, "FAIL");
typedef StaticAccumulateShape<_Shape>::type _CumuShape;
static_assert(_CumuShape::size == 3, "WRONG SHAPE");
static_assert(_CumuShape::values[0] == 20, "FAIL");
*/
template<class Shape>
class StaticShapeIndexer {
	typedef typename Shape::type I;
	typedef typename StaticAccumulateShape<Shape>::type CumulativeShape;
public:
	enum {
		NDIM = Shape::size,
		SIZE = GetIndex<CumulativeShape,0>::value * GetIndex<Shape,0>::value
	};
	__host__ __device__
	inline static IndexArray<I,NDIM> lift(I index) {
		IndexArray<I,NDIM> inds;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
		for( int d=0; d<NDIM; ++d ) {
			I cumulative_length = CumulativeShape::values[d];
			int ind = index / cumulative_length;
			index  -= ind   * cumulative_length;
			inds[d] = ind;
		}
		return inds;
	}
};

// **TODO: Generalise this to int64 indexing
//           Probably need to generalise int_fastdiv too

// WARNING: This is used as part of a HACK where it is cast to the same type
//            with a smaller NDIM. Must be very careful with its data layout.
//            Specifically, _cumu_shape[NDIM] must be stored last.
template<int NDIM>
class ShapeIndexer {
	int         _nelement;
	int         _ndim;
	int_fastdiv _cumu_shape[NDIM];
public:
	explicit inline ShapeIndexer(int const shape[NDIM], int ndim=NDIM) {
		if( ndim == 0 ) {
			// Small HACK to deal with special case of ndim==0
			_ndim = 1;
			_cumu_shape[0] = 1;
			_nelement = 1;
			return;
		}
		_ndim = ndim;
		_cumu_shape[ndim-1] = 1;
		for( int d=ndim-1; d-->0; ) {
			_cumu_shape[d] = _cumu_shape[d+1] * shape[d+1];
		}
		_nelement = _cumu_shape[0] * shape[0];
	}
	__host__ __device__
	inline IndexArray<int,NDIM> at(int i, int ndim=-1) const {
		if( ndim == -1 ) {
			ndim = _ndim;
		}
		IndexArray<int,NDIM> inds;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
		for( int d=0; d<ndim; ++d ) {
			int ind =   i / _cumu_shape[d];
			i      -= ind * _cumu_shape[d];
			inds[d] = ind;
		}
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
		for( int d=ndim; d<NDIM; ++d ) {
			inds[d] = 0;
		}
		return inds;
	}
	__host__ __device__
	inline IndexArray<int,NDIM> operator[](int i) const {
		return this->at(i, NDIM);
	}
	__host__ __device__
	inline int size() const { return _nelement; }
};
