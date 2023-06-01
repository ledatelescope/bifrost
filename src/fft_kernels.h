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
#include <hip/hip_runtime.h>

#include <hip/hip_runtime.h>


#include <bifrost/common.h>
#include <bifrost/array.h>
#include "int_fastdiv.h"

#include <hipfft/hipfft.h>
#include <hipfft/hipfftXt.h>

inline const char* _hipfftGetErrorString(hipfftResult status) {
#define DEFINE_CUFFT_RESULT_CASE(x) case x: return #x
	switch( status ) {
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_SUCCESS);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_INVALID_PLAN);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_ALLOC_FAILED);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_INVALID_TYPE);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_INVALID_VALUE);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_INTERNAL_ERROR);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_EXEC_FAILED);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_SETUP_FAILED);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_INVALID_SIZE);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_UNALIGNED_DATA);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_INCOMPLETE_PARAMETER_LIST);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_INVALID_DEVICE);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_PARSE_ERROR);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_NO_WORKSPACE);
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_NOT_IMPLEMENTED);
	// DEFINE_CUFFT_RESULT_CASE(CUFFT_LICENSE_ERROR);
#if CUDA_VERSION >= 7500
	DEFINE_CUFFT_RESULT_CASE(HIPFFT_NOT_SUPPORTED);
#endif
	default: return "Unknown CUBLAS error";
	}
#undef DEFINE_CUFFT_RESULT_CASE
}

inline BFstatus bifrost_status(hipfftResult status) {
	switch(status) {
	case HIPFFT_SUCCESS:          return BF_STATUS_SUCCESS;
	case HIPFFT_ALLOC_FAILED:     return BF_STATUS_MEM_ALLOC_FAILED;
	case HIPFFT_EXEC_FAILED:      return BF_STATUS_DEVICE_ERROR;
	case HIPFFT_NOT_IMPLEMENTED:  return BF_STATUS_UNSUPPORTED;
#if CUDA_VERSION >= 7500
	case HIPFFT_NOT_SUPPORTED:    return BF_STATUS_UNSUPPORTED;
#endif
	default: return BF_STATUS_INTERNAL_ERROR;
    }
}

#define BF_CHECK_HIPFFT_EXCEPTION(call) \
	do { \
		hipfftResult hipfft_ret = call; \
		if( hipfft_ret != HIPFFT_SUCCESS ) { \
			BF_DEBUG_PRINT(_hipfftGetErrorString(hipfft_ret)); \
		} \
		BF_ASSERT_EXCEPTION(hipfft_ret == HIPFFT_SUCCESS, \
		                    bifrost_status(hipfft_ret)); \
	} while(0)

#define BF_CHECK_HIPFFT(call) \
	do { \
		hipfftResult hipfft_ret = call; \
		if( hipfft_ret != HIPFFT_SUCCESS ) { \
			BF_DEBUG_PRINT(_hipfftGetErrorString(hipfft_ret)); \
		} \
		BF_ASSERT(hipfft_ret == HIPFFT_SUCCESS, \
		          bifrost_status(hipfft_ret)); \
	} while(0)

struct __attribute__((packed,aligned(4))) CallbackData {
	int ptr_offset;
	int ndim;
	// Note: These array sizes must be at least the max supported FFT rank
	int shape[3];
	bool inverse;
	bool do_fftshift;
	int_fastdiv istrides[3]; // Note: Elements, not bytes
	int_fastdiv inembed[3];  // Note: Elements, not bytes
	void* data;
};

BFstatus set_fft_load_callback(BFdtype       dtype,
                               int           nbit,
                               hipfftHandle  handle,
                               bool          do_fftshift,
                               CallbackData* callback_data,
                               bool*         using_callback);
