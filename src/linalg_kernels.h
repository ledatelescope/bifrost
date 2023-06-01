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

#ifndef ROCM_MATHLIBS_API_USE_HIP_COMPLEX
#define ROCM_MATHLIBS_API_USE_HIP_COMPLEX 1
#endif

#include "cuda.hpp"
#include <bifrost/config.h>
#include <bifrost/array.h>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

inline const char* _hipblasGetErrorString(hipblasStatus_t status) {
	switch(status) {
	case HIPBLAS_STATUS_SUCCESS:          return "HIPBLAS_STATUS_SUCCESS";
	case HIPBLAS_STATUS_NOT_INITIALIZED:  return "HIPBLAS_STATUS_NOT_INITIALIZED";
	case HIPBLAS_STATUS_ALLOC_FAILED:     return "HIPBLAS_STATUS_ALLOC_FAILED";
	case HIPBLAS_STATUS_INVALID_VALUE:    return "HIPBLAS_STATUS_INVALID_VALUE";
	case HIPBLAS_STATUS_ARCH_MISMATCH:    return "HIPBLAS_STATUS_ARCH_MISMATCH";
	case HIPBLAS_STATUS_MAPPING_ERROR:    return "HIPBLAS_STATUS_MAPPING_ERROR";
	case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
	case HIPBLAS_STATUS_INTERNAL_ERROR:   return "HIPBLAS_STATUS_INTERNAL_ERROR";
	case HIPBLAS_STATUS_NOT_SUPPORTED:    return "HIPBLAS_STATUS_NOT_SUPPORTED";
	case HIPBLAS_STATUS_UNKNOWN:    return "HIPBLAS_STATUS_UNKNOWN";
	default: return "Unknown CUBLAS error";
	}
}

inline BFstatus bifrost_status(hipblasStatus_t status) {
	switch(status) {
	case HIPBLAS_STATUS_SUCCESS:          return BF_STATUS_SUCCESS;
	case HIPBLAS_STATUS_NOT_INITIALIZED:  return BF_STATUS_DEVICE_ERROR;
	case HIPBLAS_STATUS_ALLOC_FAILED:     return BF_STATUS_MEM_ALLOC_FAILED;
	case HIPBLAS_STATUS_INVALID_VALUE:    return BF_STATUS_INTERNAL_ERROR;
	case HIPBLAS_STATUS_ARCH_MISMATCH:    return BF_STATUS_INTERNAL_ERROR;
	case HIPBLAS_STATUS_MAPPING_ERROR:    return BF_STATUS_DEVICE_ERROR;
	case HIPBLAS_STATUS_EXECUTION_FAILED: return BF_STATUS_DEVICE_ERROR;
	case HIPBLAS_STATUS_INTERNAL_ERROR:   return BF_STATUS_DEVICE_ERROR;
	case HIPBLAS_STATUS_NOT_SUPPORTED:    return BF_STATUS_UNSUPPORTED;
	case HIPBLAS_STATUS_UNKNOWN:    return BF_STATUS_DEVICE_ERROR;
	default: return BF_STATUS_INTERNAL_ERROR;
    }
}

#define BF_CHECK_HIPBLAS_EXCEPTION(call) \
	do { \
		hipblasStatus_t hipblas_ret = call; \
		if( hipblas_ret != HIPBLAS_STATUS_SUCCESS ) { \
			BF_DEBUG_PRINT(_hipblasGetErrorString(hipblas_ret)); \
		} \
		BF_ASSERT_EXCEPTION(hipblas_ret == HIPBLAS_STATUS_SUCCESS, \
		                    bifrost_status(hipblas_ret)); \
	} while(0)

#define BF_CHECK_HIPBLAS(call) \
	do { \
		hipblasStatus_t hipblas_ret = call; \
		if( hipblas_ret != HIPBLAS_STATUS_SUCCESS ) { \
			BF_DEBUG_PRINT(_hipblasGetErrorString(hipblas_ret)); \
		} \
		BF_ASSERT(hipblas_ret == HIPBLAS_STATUS_SUCCESS, \
		          bifrost_status(hipblas_ret)); \
	} while(0)

void bf_cherk_N(int N, int K, int nbatch,
                float alpha,
                void const* A_ptr,
                BFdtype A_type,
                int A_stride,
                int A_batchstride,
                float beta,
                void* C_ptr,
                BFdtype C_type,
                int C_stride,
                int C_batchstride,
                hipStream_t stream);

void bf_cgemm_TN_smallM(int M,
                        int N,
                        int K,
                        int nbatch,
                        float alpha,
                        void const* d_A,
                        BFdtype A_type,
                        int A_stride,
                        int A_batchstride,
                        void const* d_B,
                        BFdtype B_type,
                        int B_stride,
                        int B_batchstride,
                        float beta,
                        void*       d_C,
                        BFdtype C_type,
                        int C_stride,
                        int C_batchstride,
                        hipStream_t stream);
