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

#include "IndexArray.cuh"

// TODO: It's annoying that we have to do this; any better way?
#ifndef __CUDACC__
using std::min;
using std::max;
#endif

namespace ArrayIndexer_detail {

template<bool B, class T = void>
struct enable_if {};
template<class T>
struct enable_if<true, T> { typedef T type; };

template<typename T> struct reference_type       { typedef T& type; };
template<>           struct reference_type<void> {};

template<typename T>
struct add_lvalue_reference { typedef T& type; };
template<typename T>
struct add_lvalue_reference<T&> { typedef T& type; };
template<>
struct add_lvalue_reference<void> { typedef void type; };

} // namespace ArrayIndexer_detail

template<typename T, class Shape, class Strides>
class StaticArrayIndexerBasic {
	typedef typename Shape::type I;
protected:
	T* _data;
	I  _default_offset;
	
	template<int IND_NDIM>
	inline T& at(IndexArray<I,IND_NDIM> const& inds) const {
		I offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
		for( int d=0; d<static_min<NDIM,IND_NDIM>::value; ++d ) {
			// Index dim (tail-aligned broadcasting)
			int bd = d + static_max<IND_NDIM - NDIM, 0>::value;
			/*constexpr*/ I length = Shape::values[d];
			// Note: Converts byte strides to element strides
			/*constexpr*/ I stride = Strides::values[d] / sizeof(T);
			int ind = inds[bd];
			// Reverse indexing
			ind += (ind < 0) * length;
			// Broadcasting
			offset += (length != 1) * ind*stride;
		}
		return _data[offset];
	}
	StaticArrayIndexerBasic(StaticArrayIndexerBasic const& ) = delete;
	StaticArrayIndexerBasic& operator=(StaticArrayIndexerBasic const& ) = delete;
public:
	enum { NDIM = Shape::size };
	typedef T type;
	
	template<int IND_NDIM>
	__host__ __device__
	inline StaticArrayIndexerBasic(T* data, IndexArray<I,IND_NDIM> const& default_inds)
		: _data(data) {
		_default_offset = &this->at(default_inds) - _data;
	}
	__host__ __device__
	inline static /*constexpr*/ I size() {
		// TODO: Replace this with a metafunction
		I size_ = 1;
		for( int d=0; d<NDIM; ++d ) {
			size_ *= Shape::values[d];
		}
		return size_;
	}
	// Implicit conversion to T
	__host__ __device__
	inline operator T&() const {
		//return this->operator()(_default_inds);
		return _data[_default_offset];
	}
	// Assignment with implicit conversion
	template<typename U>
	__host__ __device__
	inline StaticArrayIndexerBasic& operator=(U const& val) {
		(T&)*this = val;
		return *this;
	}
	inline T*       operator->()       { return &(T&)*this; }
	inline T const* operator->() const { return &(T&)*this; }
	inline T&       operator*()        { return  (T&)*this; }
	inline T const& operator*()  const { return  (T&)*this; }
};

template<typename T, class Shape, class Strides>
class StaticArrayIndexer : public StaticArrayIndexerBasic<T,Shape,Strides> {
	typedef StaticArrayIndexerBasic<T,Shape,Strides> super_type;
	typedef T& reference_type;
	typedef typename Shape::type I;
public:
	enum { NDIM = Shape::size };
	
	template<int IND_NDIM>
	__host__ __device__
	inline StaticArrayIndexer(T* data, IndexArray<I,IND_NDIM> const& default_inds)
		: super_type(data, default_inds) {}
	
	template<int IND_NDIM>
	__host__ __device__
	inline reference_type operator()(IndexArray<I,IND_NDIM> const& inds) const {
		return this->at(inds);
	}
	
	template<typename... Inds>
	__host__ __device__
	inline typename ArrayIndexer_detail::enable_if<
		// Note: If NDIM==1, this clashes with IndexArray-taking method above,
		//         so we just exclude this case here and let IndexArray's
		//         constructor deal with it.
		sizeof...(Inds)==NDIM && (NDIM>0),
		reference_type>::type
		operator()(Inds... inds) const {
		// TODO: Double-check that this gets fully inlined
		I inds_array[NDIM] = {inds...};
		return this->operator()(IndexArray<I,NDIM>(inds_array));
	}
	// ***TODO: Work out how to support general a(i,j,...,_,m,n,...)
	//                                                    ^
	/*
	template<typename... IndsBeg, int IND_NDIM, typename... IndsEnd>
	typename ArrayIndexer_detail::enable_if<
		(sizeof...(IndsBeg)==1) && (sizeof...(IndsEnd)==1),
		reference_type>::type
	operator()(IndsBeg... inds_beg, IndexArray<I,IND_NDIM> inds_mid_array, IndsEnd... inds_end) {
		I inds_beg_vals[] = {inds_beg...};
		IndexArray<I,sizeof...(IndsBeg)> inds_beg_array(inds_beg_vals);
		I inds_end_vals[] = {inds_end...};
		IndexArray<I,sizeof...(IndsEnd)> inds_end_array(inds_end_vals);
		auto inds_array = inds_beg_array.join(inds_mid_array).join(inds_end_array);
		return this->operator()(inds_array);
	}
	*/
	template<int IND_NDIM, typename... IndsEnd>
	typename ArrayIndexer_detail::enable_if<
		(sizeof...(IndsEnd) > 0),
		reference_type>::type
	operator()(IndexArray<I,IND_NDIM> inds_mid_array, IndsEnd... inds_end) {
		I inds_end_vals[] = {inds_end...};
		IndexArray<I,sizeof...(IndsEnd)> inds_end_array(inds_end_vals);
		auto inds_array = inds_mid_array.join(inds_end_array);
		return this->operator()(inds_array);
	}
	
	/*
	template<int... IND_NDIMS>
	reference_type
	operator()(IndexArray<I,IND_NDIMS>... inds) {
		
		return *(*this);//this->operator()(
	}
	*/
	__host__ __device__
	inline static constexpr I shape(int dim) {
		return Shape::values[dim];
	}
	__host__ __device__
	inline static IndexArray<I,NDIM> shape() {
		return Shape::values;
	}
	// Assignment with implicit conversion
	template<typename U>
	__host__ __device__
	inline StaticArrayIndexer& operator=(U const& val) {
		(T&)*this = val;
		return *this;
	}
};

// **TODO: Generalise this to int64 indexing

// WARNING: This is used as part of a HACK where it is cast to the same type
//            with a smaller NDIM. Must be very careful with its data layout.
//            Specifically, _shape_strides[NDIM] must be stored last.
// WARNING: See above comment regarding unsafe cast HACK, and be additionally
//            careful with T potentially being void on the host side.
template<int NDIM, typename T>
class ArrayIndexer {
	typedef typename ArrayIndexer_detail::add_lvalue_reference<T>::type reference_type;
	T*  _data;
	int _ndim;
	//mutable IndexArray<int,NDIM> _default_inds;
	//T* _default_value;
	int _default_offset;
	int2 _shape_strides[NDIM];
	template<class S>
	inline void set(T* data, S shape, S byte_strides, int ndim) {
		_data = data;
		_ndim = ndim;
		for( int d=0; d<ndim; ++d ) {
			_shape_strides[d].x = shape[d];
			_shape_strides[d].y = byte_strides[d];
		}
		for( int d=ndim; d<NDIM; ++d ) {
			_shape_strides[d].x = 0;
			_shape_strides[d].y = 0;
		}
	}
public:
	__host__ __device__
	inline ArrayIndexer() : _data(0) {}
	template<class S>
	__host__ __device__
	inline ArrayIndexer(T* data, S shape, S byte_strides, int ndim=NDIM) {
		this->set(data, shape, byte_strides, ndim);
	}
	//inline ArrayIndexer(BFarray const* arr) {
	//	this->set(arr->data, arr->shape, arr->strides, arr->ndim);
	//}
	// Note: We use add_lvalue_reference to avoid creating references to void
	template<int IND_NDIM>
	__host__ __device__ inline
	reference_type
	at(IndexArray<int,IND_NDIM> inds) const {
		int offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
		for( int d=0; d<static_min<NDIM,IND_NDIM>::value; ++d ) {
			// Index dim (tail-aligned broadcasting)
			int bd = d + max(IND_NDIM - NDIM, 0);
			int shape  = _shape_strides[d].x;
			// Note: Converts byte strides to element strides
			int stride = _shape_strides[d].y / sizeof(T);
			int ind = inds[bd];
			// Reverse indexing
			ind += (ind < 0) * shape;
			// Broadcasting
			offset += (shape != 1) * ind*stride;
		}
		return _data[offset];
	}
	template<int IND_NDIM>
	__host__ __device__ inline
	reference_type
	operator()(IndexArray<int,IND_NDIM> inds) const {
		return this->at(inds);
	}
	template<typename... Inds>
	__host__ __device__
	inline typename ArrayIndexer_detail::enable_if<
		// Note: If NDIM==1, this clashes with IndexArray-taking method above,
		//         so we just exclude this case here and let IndexArray's
		//         constructor deal with it.
		sizeof...(Inds)==NDIM && (NDIM>0),
		reference_type>::type
	operator()(Inds... inds) const {
		int inds_array[NDIM] = {inds...};
		return this->operator()(inds_array);
	}
	/*
	template<int IND_NDIM>
	inline void set_default_inds(IndexArray<int,IND_NDIM> inds) {
		//_default_value = &this->at(inds);
		_default_offset = &this->at(inds) - _data;
	}
	//operator reference_type() const { return this->at(_default_inds); }
	//operator reference_type() const { return *_default_value; }
	operator reference_type() const { return _data[_default_offset]; }
	*/
	__host__ __device__
	inline int shape(int dim) const {
		return _shape_strides[dim].x;
	}
	__host__ __device__
	inline IndexArray<int,NDIM> shape() const {
		IndexArray<int,NDIM> shp;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
		for( int d=0; d<NDIM; ++d ) {
			shp[d] = _shape_strides[d].x;
		}
		return shp;
	}
};
