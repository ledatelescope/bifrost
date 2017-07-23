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

#include "cuda.hpp"

#include <cublas_v2.h>

#include <bifrost/array.h>

inline const char* _cublasGetErrorString(cublasStatus_t status) {
	switch(status) {
	case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
	case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
	case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
	default: return "Unknown CUBLAS error";
	}
}

inline BFstatus bifrost_status(cublasStatus_t status) {
	switch(status) {
	case CUBLAS_STATUS_SUCCESS:          return BF_STATUS_SUCCESS;
	case CUBLAS_STATUS_NOT_INITIALIZED:  return BF_STATUS_DEVICE_ERROR;
	case CUBLAS_STATUS_ALLOC_FAILED:     return BF_STATUS_MEM_ALLOC_FAILED;
	case CUBLAS_STATUS_INVALID_VALUE:    return BF_STATUS_INTERNAL_ERROR;
	case CUBLAS_STATUS_ARCH_MISMATCH:    return BF_STATUS_INTERNAL_ERROR;
	case CUBLAS_STATUS_MAPPING_ERROR:    return BF_STATUS_DEVICE_ERROR;
	case CUBLAS_STATUS_EXECUTION_FAILED: return BF_STATUS_DEVICE_ERROR;
	case CUBLAS_STATUS_INTERNAL_ERROR:   return BF_STATUS_DEVICE_ERROR;
	case CUBLAS_STATUS_NOT_SUPPORTED:    return BF_STATUS_UNSUPPORTED;
	case CUBLAS_STATUS_LICENSE_ERROR:    return BF_STATUS_DEVICE_ERROR;
	default: return BF_STATUS_INTERNAL_ERROR;
    }
}

#define BF_CHECK_CUBLAS_EXCEPTION(call) \
	do { \
		cublasStatus_t cublas_ret = call; \
		if( cublas_ret != CUBLAS_STATUS_SUCCESS ) { \
			BF_DEBUG_PRINT(_cublasGetErrorString(cublas_ret)); \
		} \
		BF_ASSERT_EXCEPTION(cublas_ret == CUBLAS_STATUS_SUCCESS, \
		                    bifrost_status(cublas_ret)); \
	} while(0)

#define BF_CHECK_CUBLAS(call) \
	do { \
		cublasStatus_t cublas_ret = call; \
		if( cublas_ret != CUBLAS_STATUS_SUCCESS ) { \
			BF_DEBUG_PRINT(_cublasGetErrorString(cublas_ret)); \
		} \
		BF_ASSERT(cublas_ret == CUBLAS_STATUS_SUCCESS, \
		          bifrost_status(cublas_ret)); \
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
                cudaStream_t stream);

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
                        cudaStream_t stream);
