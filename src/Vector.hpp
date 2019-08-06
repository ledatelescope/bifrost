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

#if defined __CUDACC_VER_MAJOR__ && __CUDACC_VER_MAJOR__ < 9

#define COUNT_TRAILING_ZEROS(x) \
	(32 \
	 -  1 * bool((unsigned(x) & unsigned(-signed(x))) & 0xFFFFFFFF) \
	 - 16 * bool((unsigned(x) & unsigned(-signed(x))) & 0x0000FFFF) \
	 -  8 * bool((unsigned(x) & unsigned(-signed(x))) & 0x00FF00FF) \
	 -  4 * bool((unsigned(x) & unsigned(-signed(x))) & 0x0F0F0F0F) \
	 -  2 * bool((unsigned(x) & unsigned(-signed(x))) & 0x33333333) \
	 -  1 * bool((unsigned(x) & unsigned(-signed(x))) & 0x55555555))
#define LARGEST_POW2_FACTOR(x) \
	(1 << COUNT_TRAILING_ZEROS(x))

#else // CUDA 9+

template<int N>
struct CountTrailingZeros {
	enum {
		value =
		(32
		 -  1 * bool((unsigned(N) & unsigned(-signed(N))) & 0xFFFFFFFF)
		 - 16 * bool((unsigned(N) & unsigned(-signed(N))) & 0x0000FFFF)
		 -  8 * bool((unsigned(N) & unsigned(-signed(N))) & 0x00FF00FF)
		 -  4 * bool((unsigned(N) & unsigned(-signed(N))) & 0x0F0F0F0F)
		 -  2 * bool((unsigned(N) & unsigned(-signed(N))) & 0x33333333)
		 -  1 * bool((unsigned(N) & unsigned(-signed(N))) & 0x55555555))
	};
};
template<int N>
struct LargestPow2Factor {
	enum { value = 1 << CountTrailingZeros<N>::value };
};
#define LARGEST_POW2_FACTOR(x) LargestPow2Factor<x>::value

#endif

template<typename T, int N>
class __attribute__((aligned( LARGEST_POW2_FACTOR(sizeof(T)*N) ))) Vector {
	T _v[N];
public:
	typedef T value_type;
	template<typename... U>
	Vector(U... u) : _v{u...} {}
	__host__ __device__
	inline constexpr T const& operator[](int i) const {
		return _v[i];
	}
	__host__ __device__
	inline T& operator[](int i) {
		return _v[i];
	}
};

#undef LARGEST_POW2_FACTOR
#undef COUNT_TRAILING_ZEROS

template<int NBYTE> struct StorageType;
template<> struct StorageType< 1> { typedef char  type; };
template<> struct StorageType< 2> { typedef short type; };
template<> struct StorageType< 4> { typedef int   type; };
template<> struct StorageType< 8> { typedef int2  type; };
template<> struct StorageType<16> { typedef int4  type; };
