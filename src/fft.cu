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

/*! \file fft.cu
 *  \brief This file wraps CUFFT functionality into the Bifrost C++ API.
 */

/*
  TODO: Implicitly padded/cropped transforms using load callback
        Optional fftshift of output
          Cyclic shift along each axis of n//2 elements
            (or negative shift for ifftshift).
*/

#include <bifrost/fft.h>
#include "assert.hpp"
#include "utils.hpp"
#include "cuda.hpp"
#include "ShapeIndexer.cuh"
#include "ArrayIndexer.cuh"
#include <thrust/device_vector.h>

#include <cufft.h>
#include <cufftXt.h>

const char* _cufftGetErrorString(cufftResult status) {
#define DEFINE_CUFFT_RESULT_CASE(x) case x: return #x
	switch( status ) {
	DEFINE_CUFFT_RESULT_CASE(CUFFT_SUCCESS);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_INVALID_PLAN);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_ALLOC_FAILED);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_INVALID_TYPE);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_INVALID_VALUE);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_INTERNAL_ERROR);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_EXEC_FAILED);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_SETUP_FAILED);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_INVALID_SIZE);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_UNALIGNED_DATA);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_INCOMPLETE_PARAMETER_LIST);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_INVALID_DEVICE);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_PARSE_ERROR);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_NO_WORKSPACE);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_NOT_IMPLEMENTED);
	DEFINE_CUFFT_RESULT_CASE(CUFFT_LICENSE_ERROR);
#if CUDA_VERSION >= 7500
	DEFINE_CUFFT_RESULT_CASE(CUFFT_NOT_SUPPORTED);
#endif
	default: return "Unknown CUBLAS error";
	}
#undef DEFINE_CUFFT_RESULT_CASE
}

BFstatus bifrost_status(cufftResult status) {
	switch(status) {
	case CUFFT_SUCCESS:          return BF_STATUS_SUCCESS;
	case CUFFT_ALLOC_FAILED:     return BF_STATUS_MEM_ALLOC_FAILED;
	case CUFFT_EXEC_FAILED:      return BF_STATUS_DEVICE_ERROR;
	case CUFFT_NOT_IMPLEMENTED:  return BF_STATUS_UNSUPPORTED;
#if CUDA_VERSION >= 7500
	case CUFFT_NOT_SUPPORTED:    return BF_STATUS_UNSUPPORTED;
#endif
	default: return BF_STATUS_INTERNAL_ERROR;
    }
}

#define BF_CHECK_CUFFT_EXCEPTION(call) \
	do { \
		cufftResult cufft_ret = call; \
		if( cufft_ret != CUFFT_SUCCESS ) { \
			BF_DEBUG_PRINT(_cufftGetErrorString(cufft_ret)); \
		} \
		BF_ASSERT_EXCEPTION(cufft_ret == CUFFT_SUCCESS, \
		                    bifrost_status(cufft_ret)); \
	} while(0)

#define BF_CHECK_CUFFT(call) \
	do { \
		cufftResult cufft_ret = call; \
		if( cufft_ret != CUFFT_SUCCESS ) { \
			BF_DEBUG_PRINT(_cufftGetErrorString(cufft_ret)); \
		} \
		BF_ASSERT(cufft_ret == CUFFT_SUCCESS, \
		          bifrost_status(cufft_ret)); \
	} while(0)

struct CallbackData {
	int ptr_offset;
};

__device__
cufftComplex callback_load_ci4(void*  dataIn,
                               size_t offset,
                               void*  callerInfo,
                               void*  sharedPointer) {
	// WAR for CUFFT insisting on pointers aligned to sizeof(cufftComplex)
	CallbackData* callback_data = (CallbackData*)callerInfo;
	*(char*)&dataIn += callback_data->ptr_offset;
	
	int8_t packed = ((int8_t*)dataIn)[offset];
	int8_t real = packed & 0xF0;
	int8_t imag = packed << 4;
	return make_float2(real * (1.f/128),
	                   imag * (1.f/128));
}
__device__
cufftComplex callback_load_ci8(void*  dataIn,
                               size_t offset,
                               void*  callerInfo,
                               void*  sharedPointer) {
	// WAR for CUFFT insisting on pointers aligned to sizeof(cufftComplex)
	CallbackData* callback_data = (CallbackData*)callerInfo;
	*(char*)&dataIn += callback_data->ptr_offset;
	
	char2 val = ((char2*)dataIn)[offset];
	return make_float2(val.x * (1.f/128),
	                   val.y * (1.f/128));
}
__device__
cufftComplex callback_load_ci16(void*  dataIn,
                                size_t offset,
                                void*  callerInfo,
                                void*  sharedPointer) {
	// WAR for CUFFT insisting on pointers aligned to sizeof(cufftComplex)
	CallbackData* callback_data = (CallbackData*)callerInfo;
	*(char*)&dataIn += callback_data->ptr_offset;
	
	short2 val = ((short2*)dataIn)[offset];
	return make_float2(val.x * (1.f/32768),
	                   val.y * (1.f/32768));
}
static __device__ cufftCallbackLoadC callback_load_ci4_dptr  = callback_load_ci4;
static __device__ cufftCallbackLoadC callback_load_ci8_dptr  = callback_load_ci8;
static __device__ cufftCallbackLoadC callback_load_ci16_dptr = callback_load_ci16;

template<typename T>
struct is_signed { enum { value = (((T)(-1)) < 0) }; };

template<typename T>
__host__ __device__
inline T maxval(T x=T()) { return (1<<(sizeof(T)*8-is_signed<T>::value)) - 1; }

template<typename T>
__device__
cufftReal callback_load_real(void*  dataIn,
                             size_t offset,
                             void*  callerInfo,
                             void*  sharedPointer) {
	// WAR for CUFFT insisting on pointers aligned to sizeof(cufftComplex)
	CallbackData* callback_data = (CallbackData*)callerInfo;
	*(char*)&dataIn += callback_data->ptr_offset;
	
	T val = ((T*)dataIn)[offset];
	return val * (1.f/(maxval<T>()+1));
}
static __device__ cufftCallbackLoadR callback_load_i8_dptr  = callback_load_real<int8_t>;
static __device__ cufftCallbackLoadR callback_load_i16_dptr = callback_load_real<int16_t>;
static __device__ cufftCallbackLoadR callback_load_u8_dptr  = callback_load_real<uint8_t>;
static __device__ cufftCallbackLoadR callback_load_u16_dptr = callback_load_real<uint16_t>;

class BFfft_impl {
	cufftHandle      _handle;
	bool             _real_in;
	bool             _real_out;
	int              _nbit;
	BFdtype          _itype;
	BFdtype          _otype;
	int              _batch_shape[BF_MAX_DIMS];
	size_t           _workspace_size;
	thrust::device_vector<char> _dv_tmp_storage;
	thrust::device_vector<CallbackData> _dv_callback_data;
	
	BFstatus execute_impl(void*   idata,
	                      BFdtype itype,
	                      void*   odata,
	                      BFdtype otype,
	                      BFbool  inverse,
	                      void*   tmp_storage,
	                      size_t  tmp_storage_size);
	// No copy-assign
	BFfft_impl(BFfft_impl const& );
	BFfft_impl& operator=(BFfft_impl const& );
public:
	BFfft_impl();
	~BFfft_impl();
	BFstatus init(BFarray const* in,
	              BFarray const* out,
	              int            rank,
	              int     const* axes,
	              size_t*        tmp_storage_size);
	BFstatus execute(BFarray const* in,
	                 BFarray const* out,
	                 BFbool         inverse,
	                 void*          tmp_storage,
	                 size_t         tmp_storage_size);
};

BFfft_impl::BFfft_impl() {
	BF_CHECK_CUFFT_EXCEPTION( cufftCreate(&_handle) );
}
BFfft_impl::~BFfft_impl() {
	cufftDestroy(_handle);
}

BFstatus BFfft_impl::init(BFarray const* in,
                          BFarray const* out,
                          int            rank,
                          int     const* axes,
                          size_t*        tmp_storage_size) {
	BF_ASSERT(rank > 0 && rank <= BF_MAX_DIMS, BF_STATUS_INVALID_ARGUMENT);
	BF_ASSERT(rank <= in->ndim, BF_STATUS_INVALID_ARGUMENT);
	//BF_ASSERT(
	// TODO: More assertions...
	
	_real_in  = !BF_DTYPE_IS_COMPLEX( in->dtype);
	_real_out = !BF_DTYPE_IS_COMPLEX(out->dtype);
	
	int mutable_axes[BF_MAX_DIMS];
	for( int d=0; d<rank; ++d ) {
		// Default to last 'rank' axes
		mutable_axes[d] = axes ? axes[d] : in->ndim-rank+d;
		// Allow negative axis numbers
		if( mutable_axes[d] < 0 ) {
			mutable_axes[d] += in->ndim;
		}
	}
	axes = mutable_axes;
	for( int d=0; d<in->ndim; ++d ) {
		long ilength =  in->shape[d];
		long olength = out->shape[d];
		if( (!_real_in && !_real_out) ||
		    d != axes[rank-1] ) {
			BF_ASSERT(ilength == olength,
			          BF_STATUS_INVALID_SHAPE);
		} else if( !_real_out ) {
			// Special case for last dim of R2C transforms
			BF_ASSERT(olength == ilength/2+1,
			          BF_STATUS_INVALID_SHAPE);
		} else {
			// Special case for last dim of C2R transforms
			BF_ASSERT(ilength == olength/2+1,
			          BF_STATUS_INVALID_SHAPE);
		}
		// Initialize batch shape to data shape
		_batch_shape[d] = _real_in ? ilength : olength;
	}
	// Compute transform shape and strides
#if CUDA_VERSION >= 7500
	typedef long long int_array_type;
#else
	typedef int int_array_type;
#endif
	int_array_type   shape[BF_MAX_DIMS];
	int_array_type inembed[BF_MAX_DIMS];
	int_array_type onembed[BF_MAX_DIMS];
	for( int d=0; d<rank; ++d ) {
		long ilength =  in->shape[axes[d]];
		long olength = out->shape[axes[d]];
		shape[d] = _real_in ? ilength : olength;
		if( d > 0 ) {
			BF_ASSERT( in->strides[axes[d-1]] %  in->strides[axes[d]] == 0,
			                     BF_STATUS_UNSUPPORTED_STRIDE);
			BF_ASSERT(out->strides[axes[d-1]] % out->strides[axes[d]] == 0,
			                    BF_STATUS_UNSUPPORTED_STRIDE);
			// Note: These implicitly span the batch dims where necessary
			inembed[d] =  in->strides[axes[d-1]] /  in->strides[axes[d]];
			onembed[d] = out->strides[axes[d-1]] / out->strides[axes[d]];
		} else {
			inembed[d] =  in->shape[axes[d]];
			onembed[d] = out->shape[axes[d]];
		}
		// This is not a batch dim, so exclude it from _batch_shape
		_batch_shape[axes[d]] = 1;
	}
	int itype_nbyte = BF_DTYPE_NBYTE( in->dtype);
	int otype_nbyte = BF_DTYPE_NBYTE(out->dtype);
	int istride_bytes = in->strides[axes[rank-1]];
	BF_ASSERT(istride_bytes % itype_nbyte == 0,
	                    BF_STATUS_UNSUPPORTED_STRIDE);
	int istride = istride_bytes / itype_nbyte;
	int ostride_bytes = out->strides[axes[rank-1]];
	BF_ASSERT(ostride_bytes % otype_nbyte == 0,
	                    BF_STATUS_UNSUPPORTED_STRIDE);
	int ostride = ostride_bytes / otype_nbyte;
	
	// Use longest batch dim as cuFFT batch parameter
	int batch_dim;
	bool fastest_dim_is_batch_dim = axes[rank-1] != in->ndim-1;
	if( (_real_in || _real_out) && fastest_dim_is_batch_dim ) {
		// Set the inner dim as the kernel batch, as a WAR for CUFFT requiring
		//   complex-aligned memory.
		batch_dim = in->ndim-1;
	} else {
		// Otherwise use the largest batch dim as the kernel batch for best
		//   performance.
		batch_dim = argmax_last(_batch_shape, in->ndim);
	}
	long batch = _batch_shape[batch_dim];
	_batch_shape[batch_dim] = 1;
	long idist =  in->strides[batch_dim] / itype_nbyte;
	long odist = out->strides[batch_dim] / otype_nbyte;
	
	bool fp64 = (out->dtype == BF_DTYPE_F64 ||
	             out->dtype == BF_DTYPE_CF64);
	_nbit = fp64 ? 64 : 32;
	_itype =  in->dtype;
	_otype = out->dtype;
	cufftType type;
	if(      !_real_in && !_real_out ) { type = fp64 ? CUFFT_Z2Z : CUFFT_C2C; }
	else if(  _real_in && !_real_out ) { type = fp64 ? CUFFT_D2Z : CUFFT_R2C; }
	else if( !_real_in &&  _real_out ) { type = fp64 ? CUFFT_Z2D : CUFFT_C2R; }
	else {
		BF_FAIL("Complex input and/or output",
		        BF_STATUS_INVALID_DTYPE);
	}
	BF_CHECK_CUFFT( cufftSetAutoAllocation(_handle, false) );
#if CUDA_VERSION >= 7500
	BF_CHECK_CUFFT( cufftMakePlanMany64(_handle,
#else
	BF_CHECK_CUFFT( cufftMakePlanMany  (_handle,
#endif
	                                    rank, shape,
	                                    inembed, istride, idist,
	                                    onembed, ostride, odist,
	                                    type,
	                                    batch,
	                                    &_workspace_size) );
	
	cufftCallbackLoadC callback_load_c_hptr;
	cufftCallbackLoadR callback_load_r_hptr;
	_dv_callback_data.resize(1);
	CallbackData* callback_data = thrust::raw_pointer_cast(&_dv_callback_data[0]);
	// TODO: Try to reduce repetition here
	switch( in->dtype ) {
	case BF_DTYPE_CI4: {
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_c_hptr,
		                                    callback_load_ci4_dptr,
		                                    sizeof(cufftCallbackLoadC)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(_handle,
		                                   (void**)&callback_load_c_hptr,
		                                   CUFFT_CB_LD_COMPLEX,
		                                   (void**)&callback_data) );
		break;
	}
	case BF_DTYPE_CI8: {
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_c_hptr,
		                                    callback_load_ci8_dptr,
		                                    sizeof(cufftCallbackLoadC)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(_handle,
		                                   (void**)&callback_load_c_hptr,
		                                   CUFFT_CB_LD_COMPLEX,
		                                   (void**)&callback_data) );
		break;
	}
	case BF_DTYPE_CI16: {
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_c_hptr,
		                                    callback_load_ci16_dptr,
		                                    sizeof(cufftCallbackLoadC)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(_handle,
		                                   (void**)&callback_load_c_hptr,
		                                   CUFFT_CB_LD_COMPLEX,
		                                   (void**)&callback_data) );
		break;
	}
	case BF_DTYPE_I8: {
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_r_hptr,
		                                    callback_load_i8_dptr,
		                                    sizeof(cufftCallbackLoadR)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(_handle,
		                                   (void**)&callback_load_r_hptr,
		                                   CUFFT_CB_LD_REAL,
		                                   (void**)&callback_data) );
		break;
	}
	case BF_DTYPE_I16: {
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_r_hptr,
		                                    callback_load_i16_dptr,
		                                    sizeof(cufftCallbackLoadR)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(_handle,
		                                   (void**)&callback_load_r_hptr,
		                                   CUFFT_CB_LD_REAL,
		                                   (void**)&callback_data) );
		break;
	}
	case BF_DTYPE_U8: {
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_r_hptr,
		                                    callback_load_u8_dptr,
		                                    sizeof(cufftCallbackLoadR)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(_handle,
		                                   (void**)&callback_load_r_hptr,
		                                   CUFFT_CB_LD_REAL,
		                                   (void**)&callback_data) );
		break;
	}
	case BF_DTYPE_U16: {
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_r_hptr,
		                                    callback_load_u16_dptr,
		                                    sizeof(cufftCallbackLoadR)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(_handle,
		                                   (void**)&callback_load_r_hptr,
		                                   CUFFT_CB_LD_REAL,
		                                   (void**)&callback_data) );
		break;
	}
	case BF_DTYPE_CF32: // Fall-through
	case BF_DTYPE_F32:  {
		BF_ASSERT(_nbit == 32, BF_STATUS_INVALID_DTYPE);
		break;
	}
	case BF_DTYPE_CF64: // Fall-through
	case BF_DTYPE_F64: {
		BF_ASSERT(_nbit == 64, BF_STATUS_INVALID_DTYPE);
		break;
	}
	default: {
		BF_FAIL("Supported input data type", BF_STATUS_INVALID_DTYPE);
	}
	}
	
	if( tmp_storage_size ) {
		*tmp_storage_size = _workspace_size;
	}
	return BF_STATUS_SUCCESS;
}

BFstatus BFfft_impl::execute_impl(void*   idata,
                                  BFdtype itype,
                                  void*   odata,
                                  BFdtype otype,
                                  BFbool  inverse,
                                  void*   tmp_storage,
                                  size_t  tmp_storage_size) {
	BF_ASSERT(itype == _itype, BF_STATUS_INVALID_DTYPE);
	BF_ASSERT(otype == _otype, BF_STATUS_INVALID_DTYPE);
	if( !tmp_storage ) {
		BF_TRY(_dv_tmp_storage.resize(_workspace_size));
		tmp_storage = thrust::raw_pointer_cast(&_dv_tmp_storage[0]);
	} else {
		BF_ASSERT(tmp_storage_size >= _workspace_size,
		          BF_STATUS_INSUFFICIENT_STORAGE);
	}
	BF_CHECK_CUFFT( cufftSetWorkArea(_handle, tmp_storage) );
	
	CallbackData h_callback_data;
	// WAR for CUFFT insisting that pointer be aligned to sizeof(cufftComplex)
	int alignment = (_nbit == 32 ?
	                 sizeof(cufftComplex) :
	                 sizeof(cufftDoubleComplex));
	h_callback_data.ptr_offset = (uintptr_t)idata % sizeof(cufftComplex);
	*(char**)&idata -= h_callback_data.ptr_offset;
	CallbackData* d_callback_data = thrust::raw_pointer_cast(&_dv_callback_data[0]);
	cudaMemcpyAsync(d_callback_data, &h_callback_data, sizeof(CallbackData),
	                cudaMemcpyHostToDevice, g_cuda_stream);
	
	BF_ASSERT((uintptr_t)idata % alignment == 0, BF_STATUS_UNSUPPORTED_STRIDE);
	BF_ASSERT((uintptr_t)odata % alignment == 0, BF_STATUS_UNSUPPORTED_STRIDE);
	
	if( !_real_in && !_real_out ) {
		int direction = inverse ? CUFFT_INVERSE : CUFFT_FORWARD;
		if( _nbit == 32 ) {
			BF_CHECK_CUFFT( cufftExecC2C(_handle, (cufftComplex*)idata, (cufftComplex*)odata, direction) );
		} else if( _nbit == 64 ) {
			BF_CHECK_CUFFT( cufftExecZ2Z(_handle, (cufftDoubleComplex*)idata, (cufftDoubleComplex*)odata, direction) );
		} else {
			BF_FAIL("Supported data types", BF_STATUS_UNSUPPORTED_DTYPE);
		}
	} else if( _real_in && !_real_out ) {
		if( _nbit == 32 ) {
			BF_CHECK_CUFFT( cufftExecR2C(_handle, (cufftReal*)idata, (cufftComplex*)odata) );
		} else if( _nbit == 64 ) {
			BF_CHECK_CUFFT( cufftExecD2Z(_handle, (cufftDoubleReal*)idata, (cufftDoubleComplex*)odata) );
		} else {
			BF_FAIL("Supported data types", BF_STATUS_UNSUPPORTED_DTYPE);
		}
	} else if( !_real_in && _real_out ) {
		if( _nbit == 32 ) {
			BF_CHECK_CUFFT( cufftExecC2R(_handle, (cufftComplex*)idata, (cufftReal*)odata) );
		} else if( _nbit == 64 ) {
			BF_CHECK_CUFFT( cufftExecZ2D(_handle, (cufftDoubleComplex*)idata, (cufftDoubleReal*)odata) );
		} else {
			BF_FAIL("Supported data types", BF_STATUS_UNSUPPORTED_DTYPE);
		}
	} else {
		BF_FAIL("Valid data types", BF_STATUS_INVALID_DTYPE);
	}
	return BF_STATUS_SUCCESS;
}

BFstatus BFfft_impl::execute(BFarray const* in,
                             BFarray const* out,
                             BFbool         inverse,
                             void*          tmp_storage,
                             size_t         tmp_storage_size) {
	BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
	// TODO: More assertions
	
	cudaStream_t stream = g_cuda_stream;
	// Note: It appears that all transforms from the same plan must be executed
	//         on the same stream to avoid race conditions (use of workspace?).
	BF_CHECK_CUFFT( cufftSetStream(_handle, stream) );
	ShapeIndexer<BF_MAX_DIMS> shape_indexer(_batch_shape, in->ndim);
	for( long i=0; i<shape_indexer.size(); ++i ) {
		auto inds = shape_indexer.at(i);
		void* idata = array_get_pointer( in, inds);
		void* odata = array_get_pointer(out, inds);
		BFstatus ret = this->execute_impl(idata, in->dtype,
		                                  odata, out->dtype,
		                                  inverse,
		                                  tmp_storage, tmp_storage_size);
		if( ret != BF_STATUS_SUCCESS ) {
			return ret;
		}
	}
	return BF_STATUS_SUCCESS;
}

BFstatus bfFftCreate(BFfft* plan_ptr) {
	BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*plan_ptr = new BFfft_impl(),
	                   *plan_ptr = 0);
}
BFstatus bfFftInit(BFfft          plan,
                   BFarray const* in,
                   BFarray const* out,
                   int            rank,
                   int     const* axes,
                   size_t*        tmp_storage_size) {
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
	BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
	return plan->init(in, out, rank, axes, tmp_storage_size);
}
// in, out = complex, complex => [i]fft
// in, out = real, complex    => rfft
// in, out = complex, real    => irfft
// in, out = real, real       => ERROR
// tmp_storage_size If NULL, library will allocate storage automatically
BFstatus bfFftExecute(BFfft          plan,
                      BFarray const* in,
                      BFarray const* out,
                      BFbool         inverse,
                      void*          tmp_storage,
                      size_t         tmp_storage_size) {
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
	BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
	return plan->execute(in, out, inverse, tmp_storage, tmp_storage_size);
}
BFstatus bfFftDestroy(BFfft plan) {
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	delete plan;
	return BF_STATUS_SUCCESS;
}
