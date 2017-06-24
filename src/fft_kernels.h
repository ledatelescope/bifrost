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

#include <bifrost/common.h>
#include <bifrost/array.h>
#include "int_fastdiv.h"

#include <cufft.h>
#include <cufftXt.h>

inline const char* _cufftGetErrorString(cufftResult status) {
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

inline BFstatus bifrost_status(cufftResult status) {
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
                               cufftHandle   handle,
                               bool          do_fftshift,
                               CallbackData* callback_data,
                               bool*         using_callback);
