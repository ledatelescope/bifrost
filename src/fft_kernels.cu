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

#include "fft_kernels.h"
#include "cuda.hpp"

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

BFstatus set_fft_load_callback(BFdtype        dtype,
                               int            nbit,
                               cufftHandle    handle,
                               CallbackData*  callerInfo) {
	cufftCallbackLoadC callback_load_c_hptr;
	cufftCallbackLoadR callback_load_r_hptr;
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
	case BF_DTYPE_CF32: // Fall-through
	case BF_DTYPE_F32:  {
		BF_ASSERT(nbit == 32, BF_STATUS_INVALID_DTYPE);
		break;
	}
	case BF_DTYPE_CF64: // Fall-through
	case BF_DTYPE_F64: {
		BF_ASSERT(nbit == 64, BF_STATUS_INVALID_DTYPE);
		break;
	}
	default: {
		BF_FAIL("Supported input data type", BF_STATUS_INVALID_DTYPE);
	}
	}
	return BF_STATUS_SUCCESS;
}
