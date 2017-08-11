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

/*! \file memory.h
 *  \brief Space- (host/device) aware memory management/copy/set functions
 */

#ifndef BF_MEMORY_H_INCLUDE_GUARD_
#define BF_MEMORY_H_INCLUDE_GUARD_

#include <bifrost/common.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef BF_ALIGNMENT
  #define BF_ALIGNMENT 4096//512
#endif

typedef enum BFspace_ {
	BF_SPACE_AUTO         = 0,
	BF_SPACE_SYSTEM       = 1, // aligned_alloc
	BF_SPACE_CUDA         = 2, // cudaMalloc
	BF_SPACE_CUDA_HOST    = 3, // cudaHostAlloc
	BF_SPACE_CUDA_MANAGED = 4  // cudaMallocManaged
} BFspace;

BFstatus bfMalloc(void** ptr, BFsize size, BFspace space);
BFstatus bfFree(void* ptr, BFspace space);

BFstatus bfGetSpace(const void* ptr, BFspace* space);

// Note: This is sync wrt host but async wrt device
BFstatus bfMemcpy(void*       dst,
                  BFspace     dst_space,
                  const void* src,
                  BFspace     src_space,
                  BFsize      count);
BFstatus bfMemcpy2D(void*       dst,
                    BFsize      dst_stride,
                    BFspace     dst_space,
                    const void* src,
                    BFsize      src_stride,
                    BFspace     src_space,
                    BFsize      width,
                    BFsize      height);
BFstatus bfMemset(void*   ptr,
                  BFspace space,
                  int     value,
                  BFsize  count);
BFstatus bfMemset2D(void*   ptr,
                    BFsize  stride,
                    BFspace space,
                    int     value,
                    BFsize  width,
                    BFsize  height);
BFsize bfGetAlignment();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_MEMORY_H_INCLUDE_GUARD_
