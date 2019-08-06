/*
 * Copyright (c) 2016, The Bifrost Authors. All rights reserved.
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#include <bifrost/common.h>
#include <bifrost/memory.h>
#include <bifrost/array.h>
#include "cuda.hpp"
#include "IndexArray.cuh"

#include <algorithm>
#include <stdexcept>
#include <cstring> // For ::memcpy
#include <cassert>

#define BF_DTYPE_IS_COMPLEX(dtype) bool((dtype) & BF_DTYPE_COMPLEX_BIT)
#define BF_DTYPE_VECTOR_LENGTH(dtype) \
	((((dtype) & BF_DTYPE_VECTOR_BITS) >> BF_DTYPE_VECTOR_BIT0) + 1)
#define BF_DTYPE_VECTOR_LENGTH_MAX \
	((BF_DTYPE_VECTOR_BITS + 1) >> BF_DTYPE_VECTOR_BIT0)
#define BF_DTYPE_SET_VECTOR_LENGTH(dtype, veclen) \
	BFdtype((dtype & ~BF_DTYPE_VECTOR_BITS) | \
	        ((veclen - 1) << BF_DTYPE_VECTOR_BIT0))
// **TODO: Add support for string type that encodes up to 255 bytes
#define BF_DTYPE_NBIT(dtype) \
	(((dtype) & BF_DTYPE_NBIT_BITS) * \
	 (BF_DTYPE_IS_COMPLEX(dtype)+1) * \
	 (BF_DTYPE_VECTOR_LENGTH(dtype)))
#define BF_DTYPE_NBYTE(dtype) (BF_DTYPE_NBIT(dtype)/8)

inline BFdtype same_sized_storage_dtype(BFdtype dtype) {
	int nbit = BF_DTYPE_NBIT(dtype);
	return BFdtype(nbit | BF_DTYPE_STORAGE_TYPE);
	/*
	// Returns a vector type with the same size, for efficient load/store
	int nbyte = BF_DTYPE_NBYTE(dtype);
	int basetype;
	if(      nbyte % 4 == 0 ) basetype = BF_DTYPE_U32;
	else if( nbyte % 2 == 0 ) basetype = BF_DTYPE_U16;
	else                      basetype = BF_DTYPE_U8;
	int veclen = nbyte / BF_DTYPE_NBYTE(basetype);
	assert(veclen <= BF_DTYPE_VECTOR_LENGTH_MAX);
	return BF_DTYPE_SET_VECTOR_LENGTH(basetype, veclen);
	*/
}

// Note: Does not support in-place execution
inline void invert_permutation(int ndim,
                               int const* in,
                               int*       out) {
	for( int d=0; d<ndim; ++d ) {
		out[in[d]] = d;
	}
}

inline void merge_last_dim_into_dtype(BFarray const* in,
                                      BFarray*       out) {
	::memcpy(out, in, sizeof(BFarray));
	int ndim = in->ndim;
	int veclen = BF_DTYPE_VECTOR_LENGTH(in->dtype);
	veclen *= in->shape[ndim-1];
	assert(veclen <= BF_DTYPE_VECTOR_LENGTH_MAX);
	out->dtype = BF_DTYPE_SET_VECTOR_LENGTH(out->dtype, veclen);
	--(out->ndim);
}

// TODO: Check that these wrap/overflow properly
inline BFoffset round_up(BFoffset val, BFoffset mult) {
	return (val == 0 ?
	        0 :
	        ((val-1)/mult+1)*mult);
}
inline BFoffset round_up_pow2(BFoffset a) {
    size_t r = a-1;
    for( int i=1; i<=(int)sizeof(BFoffset)*8/2; i<<=1 ) r |= r >> i;
    return r+1;
}
inline BFoffset div_round_up(BFoffset n, BFoffset d) {
	return (n == 0 ?
	        0 :
	        (n-1)/d+1);
}

template<typename T>
inline T gcd(T u, T v) {
	return (v == 0) ? u : gcd(v, u % v);
}

template<typename T>
T div_up(T n, T d) {
	return (n-1)/d+1;
}
template<typename T>
bool is_pow2(T v) {
	return v && !(v & (v - 1));
}
template<typename T>
T ilog2(T v) {
	T r;
	T shift;
	r =     (v > 0xFFFFFFFF) << 5; v >>= r;
	shift = (v > 0xFFFF    ) << 4; v >>= shift; r |= shift;
	shift = (v > 0xFF      ) << 3; v >>= shift; r |= shift;
	shift = (v > 0xF       ) << 2; v >>= shift; r |= shift;
	shift = (v > 0x3       ) << 1; v >>= shift; r |= shift;
	                                            r |= (v >> 1);
	return r;

}

inline bool is_big_endian() {
	union {
		uint32_t i;
		int8_t   c[4];
	} ic = {0x01020304};
	return (ic.c[0] == 1);
}

inline void byteswap_impl(uint64_t value, uint64_t* result) {
	*result =
		((value & 0xFF00000000000000u) >> 56u) |
		((value & 0x00FF000000000000u) >> 40u) |
		((value & 0x0000FF0000000000u) >> 24u) |
		((value & 0x000000FF00000000u) >>  8u) |
		((value & 0x00000000FF000000u) <<  8u) |
		((value & 0x0000000000FF0000u) << 24u) |
		((value & 0x000000000000FF00u) << 40u) |
		((value & 0x00000000000000FFu) << 56u);
}
inline void byteswap_impl(uint32_t value, uint32_t* result) {
	*result =
		((value & 0xFF000000u) >> 24u) |
		((value & 0x00FF0000u) >>  8u) |
		((value & 0x0000FF00u) <<  8u) |
		((value & 0x000000FFu) << 24u);
}
inline void byteswap_impl(uint16_t value, uint16_t* result) {
	*result =
		((value & 0xFF00u) >> 8u) |
		((value & 0x00FFu) << 8u);
}

template<typename T, typename U>
#if BF_CUDA_ENABLED
__host__ __device__
#endif
inline T type_pun(U x) {
	static_assert(sizeof(T) == sizeof(U),
	              "Cannot pun type to different size");
	union punner_union {
		T t;
		U u;
		__host__ __device__ inline punner_union() {}
	} punner;
	punner.u = x;
	return punner.t;
}

template<typename T>
inline typename std::enable_if<sizeof(T)==8>::type
byteswap(T value, T* result) {
	return byteswap_impl(type_pun<uint64_t>(value),
	                     (uint64_t*)result);
}
template<typename T>
inline typename std::enable_if<sizeof(T)==4>::type
byteswap(T value, T* result) {
	return byteswap_impl(type_pun<uint32_t>(value),
	                     (uint32_t*)result);
}
template<typename T>
inline typename std::enable_if<sizeof(T)==2>::type
byteswap(T value, T* result) {
	return byteswap_impl(type_pun<uint16_t>(value),
	                     (uint16_t*)result);
}
template<typename T>
inline typename std::enable_if<sizeof(T)==1>::type
byteswap(T value, T* result) {
	*result = value;
}

inline BFbool space_accessible_from(BFspace space, BFspace from) {
#if !defined BF_CUDA_ENABLED || !BF_CUDA_ENABLED
	return space == BF_SPACE_SYSTEM;
#else
	switch( from ) {
	case BF_SPACE_SYSTEM: return (space == BF_SPACE_SYSTEM ||
	                              space == BF_SPACE_CUDA_HOST ||
	                              space == BF_SPACE_CUDA_MANAGED);
	case BF_SPACE_CUDA:   return (space == BF_SPACE_CUDA ||
	                              space == BF_SPACE_CUDA_MANAGED);
	// TODO: Need to use something else here?
	default: throw std::runtime_error("Internal error");
	}
#endif
}

inline bool shapes_equal(const BFarray* a,
                         const BFarray* b) {
	if( a->ndim != b->ndim ) {
		return false;
	}
	for( int d=0; d<a->ndim; ++d ) {
		if( a->shape[d] != b->shape[d] ) {
			return false;
		}
	}
	return true;
}

inline long shape_size(int ndim, const long shape[BF_MAX_DIMS]) {
	long size = 1;
	for( int d=0; d<ndim; ++d ) {
		size *= shape[d];
	}
	return size;
}

inline BFsize capacity_bytes(const BFarray* array) {
	return array->strides[0] * array->shape[0];
}
inline bool is_contiguous(const BFarray* array) {
	// TODO: Consider supporting ndim=0 (scalar arrays)
	//if( array->ndim == 0 ) {
	//	return true;
	//}
	BFsize logical_size = BF_DTYPE_NBYTE(array->dtype);
	for( int d=0; d<array->ndim; ++d ) {
		logical_size *= array->shape[d];
	}
	BFsize physical_size = capacity_bytes(array);
	return logical_size == physical_size;
}
inline BFsize num_contiguous_elements(const BFarray* array ) {
	// TODO: Consider supporting ndim=0 (scalar arrays)
	//if( array->ndim == 0 ) {
	//	return 1;
	//}
	// Assumes array is contiguous
	return capacity_bytes(array) / BF_DTYPE_NBYTE(array->dtype);
}

// Merges together contiguous dimensions
//   Copies (shallow) 'in' to 'out' and writes new ndim, shape, and strides
/*
inline void squeeze_contiguous_dims(BFarray const* in,
                                    BFarray*       out,
                                    unsigned long  keep_mask=0) {
	::memcpy(out, in, sizeof(BFarray));
	int odim = 0;
	int32_t osize = 1;
	for( int idim=0; idim<in->ndim; ++idim ) {
		osize *= in->shape[idim];
		int32_t logical_stride = in->strides[idim+1]*in->shape[idim+1];
		bool is_padded_dim = (in->strides[idim] != logical_stride);
		bool is_last_dim = (idim == in->ndim-1);
		bool is_keep_dim = keep_mask & (1<<idim);
		if( is_last_dim || is_padded_dim || is_keep_dim ) {
			out->shape[odim]   = osize;
			out->strides[odim] = in->strides[idim];
			osize = 1;
			++odim;
		}
	}
	out->ndim = odim;
}
*/
inline void print_array(BFarray const* a, const char* name=0) {
	std::cout << "Array<(";
	if( name ) {
		std::cout << "name=" << name << ", ";
	}
	std::cout << "shape=(";
	for( int d=0; d<a->ndim; ++d ) {
		std::cout << a->shape[d] << ",";
	}
	std::cout << "), strides=(";
	for( int d=0; d<a->ndim; ++d ) {
		std::cout << a->strides[d] << ",";
	}
	std::cout << ")>" << std::endl;
}
// Supports in == out
inline void remove_dim(BFarray const* in,
                       BFarray*       out,
                       int            dim) {
	::memcpy(out, in, sizeof(BFarray));
	for( int d=dim; d<in->ndim; ++d ) {
		out->shape[d]   = in->shape[d+1];
		out->strides[d] = in->strides[d+1];
	}
	--(out->ndim);
}
// Supports in == out
inline void split_dim(BFarray const* in,
                      BFarray*       out,
                      int            dim,
                      int            n) {
	::memcpy(out, in, sizeof(BFarray));
	for( int d=in->ndim; d-->dim+1; ) {
		out->shape[d+1]   = in->shape[d];
		out->strides[d+1] = in->strides[d];
	}
	out->shape[dim+1]   = n;
	out->shape[dim]     = in->shape[dim] / n;
	out->strides[dim+1] = in->strides[dim];
	out->strides[dim]   = in->strides[dim] * n;
	++out->ndim;
}
// Merges together dimensions, keeping only those with the corresponding bit
//   set in keep_mask.
inline void flatten(BFarray const* in,
                    BFarray*       out,
                    unsigned long  keep_mask=0) {
	::memcpy(out, in, sizeof(BFarray));
	int odim = 0;
	long osize = 1;
	for( int idim=0; idim<in->ndim; ++idim ) {
		osize *= in->shape[idim];
		bool is_last_dim = (idim == in->ndim-1);
		bool is_keep_dim = keep_mask & (1ul<<idim);
		if( is_last_dim || is_keep_dim ) {
			out->shape[odim]   = osize;
			out->strides[odim] = in->strides[idim];
			osize = 1;
			++odim;
		}
	}
	out->ndim = odim;
}
inline void flatten_shape(int*          ndim_ptr,
                          long          shape[BF_MAX_DIMS],
                          unsigned long keep_mask=0) {
	int ndim = *ndim_ptr;
	int odim = 0;
	long osize = 1;
	for( int idim=0; idim<ndim; ++idim ) {
		osize *= shape[idim];
		bool is_last_dim = (idim == ndim-1);
		bool is_keep_dim = keep_mask & (1ul<<idim);
		if( is_last_dim || is_keep_dim ) {
			shape[odim++] = osize;
			osize = 1;
		}
	}
	*ndim_ptr = odim;
}
/*
inline void flatten_contiguous_dims(BFarray const* in,
                                    BFarray*       out,
                                    unsigned long  keep_mask=0) {
	// Note: Need to keep the dim _before_ the padded dim, hence the >> 1
	keep_mask |= padded_dims_mask(in) >> 1;
	return flatten(in, out, keep_mask);
}
*/
// Returns mask where a 1-bit indicates that the next corresponding dim is
//   padded.
// Note: Uses next dim instead of actual because this is what's needed
//         for flatten().
inline unsigned long padded_dims_mask(BFarray const* arr) {
	unsigned long mask = 0;
	// Note: Padding does not make sense for first dim
	for( int d=1; d<arr->ndim; ++d ) {
		long logical_stride  = arr->strides[d]*arr->shape[d];
		long physical_stride = arr->strides[d-1];
		bool padded = (logical_stride != physical_stride);
		mask |= ((unsigned long)padded) << (d-1);
	}
	return mask;
}
// Returns mask where a 1-bit indicates that the corresponding dim must be
//   broadcast to match shape.
inline unsigned long broadcast_dims_mask(BFarray const* arr,
                                         int            ndim,
                                         const long     shape[BF_MAX_DIMS]) {
	unsigned long mask = 0;
	for( int d=0; d<std::min(ndim, arr->ndim); ++d ) {
		long length = arr->shape[d];
		// Index dim (tail-aligned broadcasting)
		int bd = d + std::max(ndim - arr->ndim, 0);
		if( length != shape[bd] ) {
			if( length == 1 ) {
				mask |= 1ul << bd;
			} else {
				// Unbroadcastable
				return (unsigned long)-1; // TODO: Should really report error somehow
			}
		}
	}
	return mask;
}

inline bool broadcast_shapes(int  narray,
                             BFarray const*const* arrays,
                             long shape[BF_MAX_DIMS],
                             int* ndim_ptr) {
	int ndim = 0;
	for( int i=0; i<narray; ++i ) {
		// Find number of broadcast dims
		ndim = std::max(ndim, arrays[i]->ndim);
	}
	for( int bd=0; bd<ndim; ++bd ) {
		// Initialize all broadcast dims to 1
		shape[bd] = 1;
	}
	//std::cout << "NDIM = " << ndim << std::endl;
	for( int i=0; i<narray; ++i ) {
		for( int d=0; d<arrays[i]->ndim; ++d ) {
			// Dims are right-aligned relative to broadcast shape
			int bd = d + (ndim - arrays[i]->ndim);
			long& blength = shape[bd];
			long   length = arrays[i]->shape[d];
			if( length < 1 ) {
				// Invalid array shape
				//std::cout << "INVALID " << length << std::endl;
				return false;
			} else if( blength == 1 ) {
				// Initialize this broadcast dim
				blength = length;
			} else if( length != 1 && length != blength ) {
				// Cannot be broadcast
				//std::cout << "NOBROADCAST " << length << " != " << blength << std::endl;
				return false;
			}
		}
	}
	*ndim_ptr = ndim;
	return true;
}

template<typename I>
inline void* array_get_pointer(BFarray const* arr,
                               IndexArray<I,BF_MAX_DIMS> const& inds,
                               int ndim=-1) {
	if( ndim == -1 ) {
		ndim = arr->ndim;
	}
	I offset = 0;
	for( int d=0; d<ndim; ++d ) {
		I length = arr->shape[d];
		I stride = arr->strides[d];
		if( length == 1 ) {
			// Broadcasting
			continue;
		}
		I ind = inds[d];
		if( ind < 0 ) {
			// Reverse indexing
			ind += length;
		}
		offset += ind*stride;
	}
	return &((char*)arr->data)[offset];
}

template<typename T>
inline size_t argmax_first(T const* arr, size_t n) {
	return std::max_element(arr, arr+n) - arr;
}
template<typename T>
inline size_t argmax_last(T const* arr, size_t n) {
	std::reverse_iterator<T const*> beg(arr+n);
	return n-1 - (std::max_element(beg, beg+n) - beg);
}

template<int NBIT, typename ConvertType=float, typename AccessType=char>
struct NbitReader {
	typedef ConvertType value_type;
	enum { MASK = (1<<NBIT)-1 };
	AccessType const* __restrict__ data;
	NbitReader(void* data_) : data((AccessType*)data_) {}
#if BF_CUDA_ENABLED
	__host__ __device__
#endif
	inline ConvertType operator[](int n) const {
		// TODO: Beware of overflow here
		AccessType word = data[n * NBIT / (sizeof(AccessType)*8)];
		int k = n % ((sizeof(AccessType)*8) / NBIT);
		return (word >> (k*NBIT)) & MASK;
	}
	inline ConvertType operator*() const {
		return (*this)[0];
	}
};

template<typename T> struct value_type           { typedef typename T::value_type type; };
template<typename T> struct value_type<T*>       { typedef T       type; };
template<typename T> struct value_type<T const*> { typedef T const type; };
