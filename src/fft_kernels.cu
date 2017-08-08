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

#include "fft_kernels.h"
#include "cuda.hpp"

__device__
inline size_t pre_fftshift(size_t        offset,
                           CallbackData* cb) {
	// For inverse transforms with apply_fftshift=true, we cyclically shift
	//   the input data here by modifying the read offset.
	if( cb->do_fftshift && cb->inverse ) {
		for( int d=0; d<cb->ndim; ++d ) {
			// Compute the index of this element along dimension d
			// **TODO: 64-bit indexing support
			int  size    = cb->shape[d];
			auto stride  = cb->istrides[d];
			auto inembed = cb->inembed[d];
			int i = (int)offset / stride % inembed;
			int shift = (i < (size-size/2)) ? size/2 : -(size-size/2);
			offset += shift*stride;
		}
	}
	return offset;
}
template<typename Complex>
__device__
inline Complex post_fftshift(size_t        offset,
                             Complex       value,
                             CallbackData* cb) {
	// For forward transforms with apply_fftshift=true, we cyclically shift
	//   the output data by phase-rotating the input data here.
	if( cb->do_fftshift && !cb->inverse ) {
		for( int d=0; d<cb->ndim; ++d ) {
			// Compute the index of this element along dimension d
			// **TODO: 64-bit indexing support
			int  size    = cb->shape[d];
			auto stride  = cb->istrides[d];
			auto inembed = cb->inembed[d];
			int i = (int)offset / stride % inembed;
			// We achieve a cyclic shift of the FFT output (aka fftshift)
			//   by multiplying the input by a phase shift.
			//   Note that this only works for complex input
			if( size % 2 == 0 ) {
				// For even sizes, the phase multiplication reduces to a
				// simple form: {even i: +1, odd i: -1}.
				if( i%2 ) {
					value.x = -value.x;
					value.y = -value.y;
				}
			} else {
				// For odd sizes we must do the math in full
				// TODO: Confirm that float and __sincosf provide enough
				//         precision for all practical FFT sizes.
				const float pi = 3.1415926535897932;
				float phase = 2*pi*i/size*(size/2);
				float sin_phase, cos_phase;
				__sincosf(phase, &sin_phase, &cos_phase);
				Complex tmp;
				tmp.x  = value.x*cos_phase;
				tmp.x -= value.y*sin_phase;
				tmp.y  = value.x*sin_phase;
				tmp.y += value.y*cos_phase;
				value = tmp;
			}
		}
	}
	return value;
}
__device__
cufftComplex callback_load_ci4(void*  dataIn,
                               size_t offset,
                               void*  callerInfo,
                               void*  sharedPointer) {
	// WAR for CUFFT insisting on pointers aligned to sizeof(cufftComplex)
	CallbackData* callback_data = (CallbackData*)callerInfo;
	*(char**)&dataIn += callback_data->ptr_offset;
	offset = pre_fftshift(offset, callback_data);
	int8_t packed = ((int8_t*)dataIn)[offset];
	int8_t real = packed & 0xF0;
	int8_t imag = packed << 4;
	cufftComplex result = make_float2(real * (1.f/128),
	                                  imag * (1.f/128));
	result = post_fftshift(offset, result, callback_data);
	return result;
}
__device__
cufftComplex callback_load_ci8(void*  dataIn,
                               size_t offset,
                               void*  callerInfo,
                               void*  sharedPointer) {
	// WAR for CUFFT insisting on pointers aligned to sizeof(cufftComplex)
	CallbackData* callback_data = (CallbackData*)callerInfo;
	*(char**)&dataIn += callback_data->ptr_offset;
	offset = pre_fftshift(offset, callback_data);
	char2 val = ((char2*)dataIn)[offset];
	cufftComplex result = make_float2(val.x * (1.f/128),
	                                  val.y * (1.f/128));
	result = post_fftshift(offset, result, callback_data);
	return result;
}

__device__
cufftComplex callback_load_ci16(void*  dataIn,
                                size_t offset,
                                void*  callerInfo,
                                void*  sharedPointer) {
	// WAR for CUFFT insisting on pointers aligned to sizeof(cufftComplex)
	CallbackData* callback_data = (CallbackData*)callerInfo;
	*(char**)&dataIn += callback_data->ptr_offset;
	offset = pre_fftshift(offset, callback_data);
	short2 val = ((short2*)dataIn)[offset];
	cufftComplex result = make_float2(val.x * (1.f/32768),
	                                  val.y * (1.f/32768));
	result = post_fftshift(offset, result, callback_data);
	return result;
}
__device__
cufftComplex callback_load_cf32(void*  dataIn,
                                size_t offset,
                                void*  callerInfo,
                                void*  sharedPointer) {
	CallbackData* callback_data = (CallbackData*)callerInfo;
	// Note: cufftComplex loads must be aligned
	offset = pre_fftshift(offset, callback_data);
	cufftComplex result = ((cufftComplex*)dataIn)[offset];
	result = post_fftshift(offset, result, callback_data);
	return result;
}
__device__
cufftDoubleComplex callback_load_cf64(void*  dataIn,
                                      size_t offset,
                                      void*  callerInfo,
                                      void*  sharedPointer) {
	CallbackData* callback_data = (CallbackData*)callerInfo;
	// Note: cufftDoubleComplex loads must be aligned
	offset = pre_fftshift(offset, callback_data);
	cufftDoubleComplex result = ((cufftDoubleComplex*)dataIn)[offset];
	result = post_fftshift(offset, result, callback_data);
	return result;
}
static __device__ cufftCallbackLoadC callback_load_ci4_dptr  = callback_load_ci4;
static __device__ cufftCallbackLoadC callback_load_ci8_dptr  = callback_load_ci8;
static __device__ cufftCallbackLoadC callback_load_ci16_dptr = callback_load_ci16;
static __device__ cufftCallbackLoadC callback_load_cf32_dptr = callback_load_cf32;
static __device__ cufftCallbackLoadZ callback_load_cf64_dptr = callback_load_cf64;

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
	*(char**)&dataIn += callback_data->ptr_offset;
	
	T val = ((T*)dataIn)[offset];
	cufftReal result = val * (1.f/(maxval<T>()+1));
	return result;
}
static __device__ cufftCallbackLoadR callback_load_i8_dptr  = callback_load_real<int8_t>;
static __device__ cufftCallbackLoadR callback_load_i16_dptr = callback_load_real<int16_t>;
static __device__ cufftCallbackLoadR callback_load_u8_dptr  = callback_load_real<uint8_t>;
static __device__ cufftCallbackLoadR callback_load_u16_dptr = callback_load_real<uint16_t>;

BFstatus set_fft_load_callback(BFdtype       dtype,
                               int           nbit,
                               cufftHandle   handle,
                               bool          do_fftshift,
                               CallbackData* callerInfo,
                               bool*         using_callback) {
	cufftCallbackLoadC callback_load_c_hptr;
	cufftCallbackLoadR callback_load_r_hptr;
	cufftCallbackLoadZ callback_load_z_hptr;
	*using_callback = true;
	// TODO: Try to reduce repetition here
	switch( dtype ) {
	case BF_DTYPE_CI4: {
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_c_hptr,
		                                    callback_load_ci4_dptr,
		                                    sizeof(cufftCallbackLoadC)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(handle,
		                                   (void**)&callback_load_c_hptr,
		                                   CUFFT_CB_LD_COMPLEX,
		                                   (void**)&callerInfo) );
		break;
	}
	case BF_DTYPE_CI8: {
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_c_hptr,
		                                    callback_load_ci8_dptr,
		                                    sizeof(cufftCallbackLoadC)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(handle,
		                                   (void**)&callback_load_c_hptr,
		                                   CUFFT_CB_LD_COMPLEX,
		                                   (void**)&callerInfo) );
		break;
	}
	case BF_DTYPE_CI16: {
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_c_hptr,
		                                    callback_load_ci16_dptr,
		                                    sizeof(cufftCallbackLoadC)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(handle,
		                                   (void**)&callback_load_c_hptr,
		                                   CUFFT_CB_LD_COMPLEX,
		                                   (void**)&callerInfo) );
		break;
	}
	case BF_DTYPE_I8: {
		BF_ASSERT(!do_fftshift, BF_STATUS_UNSUPPORTED);
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_r_hptr,
		                                    callback_load_i8_dptr,
		                                    sizeof(cufftCallbackLoadR)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(handle,
		                                   (void**)&callback_load_r_hptr,
		                                   CUFFT_CB_LD_REAL,
		                                   (void**)&callerInfo) );
		break;
	}
	case BF_DTYPE_I16: {
		BF_ASSERT(!do_fftshift, BF_STATUS_UNSUPPORTED);
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_r_hptr,
		                                    callback_load_i16_dptr,
		                                    sizeof(cufftCallbackLoadR)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(handle,
		                                   (void**)&callback_load_r_hptr,
		                                   CUFFT_CB_LD_REAL,
		                                   (void**)&callerInfo) );
		break;
	}
	case BF_DTYPE_U8: {
		BF_ASSERT(!do_fftshift, BF_STATUS_UNSUPPORTED);
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_r_hptr,
		                                    callback_load_u8_dptr,
		                                    sizeof(cufftCallbackLoadR)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(handle,
		                                   (void**)&callback_load_r_hptr,
		                                   CUFFT_CB_LD_REAL,
		                                   (void**)&callerInfo) );
		break;
	}
	case BF_DTYPE_U16: {
		BF_ASSERT(!do_fftshift, BF_STATUS_UNSUPPORTED);
		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_r_hptr,
		                                    callback_load_u16_dptr,
		                                    sizeof(cufftCallbackLoadR)),
		               BF_STATUS_DEVICE_ERROR );
		BF_CHECK_CUFFT( cufftXtSetCallback(handle,
		                                   (void**)&callback_load_r_hptr,
		                                   CUFFT_CB_LD_REAL,
		                                   (void**)&callerInfo) );
		break;
	}
	case BF_DTYPE_CF32: {
		if( do_fftshift ) {
			BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_c_hptr,
			                                    callback_load_cf32_dptr,
			                                    sizeof(cufftCallbackLoadC)),
			               BF_STATUS_DEVICE_ERROR );
			BF_CHECK_CUFFT( cufftXtSetCallback(handle,
			                                   (void**)&callback_load_c_hptr,
			                                   CUFFT_CB_LD_COMPLEX,
			                                   (void**)&callerInfo) );
			break;
		} else {
			// Fall-through
		}
	}
	case BF_DTYPE_F32:  {
		BF_ASSERT(nbit == 32, BF_STATUS_INVALID_DTYPE);
		BF_ASSERT(!do_fftshift, BF_STATUS_UNSUPPORTED);
		// No callback needed
		*using_callback = false;
		break;
	}
	case BF_DTYPE_CF64: {
		if( do_fftshift ) {
			BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_z_hptr,
			                                    callback_load_cf64_dptr,
			                                    sizeof(cufftCallbackLoadZ)),
			               BF_STATUS_DEVICE_ERROR );
			BF_CHECK_CUFFT( cufftXtSetCallback(handle,
			                                   (void**)&callback_load_z_hptr,
			                                   CUFFT_CB_LD_COMPLEX_DOUBLE,
			                                   (void**)&callerInfo) );
			break;
		} else {
			// Fall-through
		}
	}
	case BF_DTYPE_F64: {
		BF_ASSERT(nbit == 64, BF_STATUS_INVALID_DTYPE);
		BF_ASSERT(!do_fftshift, BF_STATUS_UNSUPPORTED);
		// No callback needed
		*using_callback = false;
		break;
	}
	default: {
		BF_FAIL("Supported input data type", BF_STATUS_INVALID_DTYPE);
	}
	}
	return BF_STATUS_SUCCESS;
}
