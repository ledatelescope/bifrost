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

#include <bifrost/memory.h>
#include "utils.hpp"
#include "cuda.hpp"
#include "trace.hpp"

#include <cstdlib> // For posix_memalign
#include <cstring> // For memcpy
#include <iostream>

#define BF_IS_POW2(x) (x) && !((x) & ((x) - 1))
static_assert(BF_IS_POW2(BF_ALIGNMENT), "BF_ALIGNMENT must be a power of 2");
#undef BF_IS_POW2
//static_assert(BF_ALIGNMENT >= 8,        "BF_ALIGNMENT must be >= 8");

BFstatus bfGetSpace(const void* ptr, BFspace* space) {
	BF_ASSERT(ptr, BF_STATUS_INVALID_POINTER);
#if !defined BF_CUDA_ENABLED || !BF_CUDA_ENABLED
	*space = BF_SPACE_SYSTEM;
#else
	cudaPointerAttributes ptr_attrs;
	cudaError_t ret = cudaPointerGetAttributes(&ptr_attrs, ptr);
	BF_ASSERT(ret == cudaSuccess || ret == cudaErrorInvalidValue,
	          BF_STATUS_DEVICE_ERROR);
	if( ret == cudaErrorInvalidValue ) {
		// TODO: Is there a better way to find out how a pointer was allocated?
		//         Alternatively, is there a way to prevent this from showing
		//           up in cuda-memcheck?
		// Note: cudaPointerGetAttributes only works for memory allocated with
		//         CUDA API functions, so if it fails we just assume sysmem.
		*space = BF_SPACE_SYSTEM;
		// WAR to avoid the ignored failure showing up later
		cudaGetLastError();
	} else if( ptr_attrs.isManaged ) {
		*space = BF_SPACE_CUDA_MANAGED;
	} else {
		switch( ptr_attrs.memoryType ) {
		case cudaMemoryTypeHost:   *space = BF_SPACE_SYSTEM; break;
		case cudaMemoryTypeDevice: *space = BF_SPACE_CUDA;   break;
		default: {
			// This should never be reached
			BF_FAIL("Valid memoryType", BF_STATUS_INTERNAL_ERROR);
		}
		}
	}
#endif
	return BF_STATUS_SUCCESS;
}

const char* bfGetSpaceString(BFspace space) {
	// TODO: Is there a better way to do this that does not involve hard 
	//       coding all of these values twice (one in memory.h for the 
	//       enum, once here)?
	
	
	switch( space ) {
		case BF_SPACE_AUTO:         return "auto";
		case BF_SPACE_SYSTEM:       return "system";
		case BF_SPACE_CUDA:         return "cuda";
		case BF_SPACE_CUDA_HOST:    return "cuda_host";
		case BF_SPACE_CUDA_MANAGED: return "cuda_managed";
		default: return "unknown";
	}
}

BFstatus bfMalloc(void** ptr, BFsize size, BFspace space) {
	//printf("bfMalloc(%p, %lu, %i)\n", ptr, size, space);
	void* data;
	switch( space ) {
	case BF_SPACE_SYSTEM: {
		//data = std::aligned_alloc(std::max(BF_ALIGNMENT,8), size);
		int err = ::posix_memalign((void**)&data, std::max(BF_ALIGNMENT,8), size);
		BF_ASSERT(!err, BF_STATUS_MEM_ALLOC_FAILED);
		//if( err ) data = nullptr;
		//printf("bfMalloc --> %p\n", data);
		break;
	}
#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
	case BF_SPACE_CUDA: {
		BF_CHECK_CUDA(cudaMalloc((void**)&data, size),
		              BF_STATUS_MEM_ALLOC_FAILED);
		break;
	}
	case BF_SPACE_CUDA_HOST: {
		unsigned flags = cudaHostAllocDefault;
		BF_CHECK_CUDA(cudaHostAlloc((void**)&data, size, flags),
		              BF_STATUS_MEM_ALLOC_FAILED);
		break;
	}
	case BF_SPACE_CUDA_MANAGED: {
		unsigned flags = cudaMemAttachGlobal;
		BF_CHECK_CUDA(cudaMallocManaged((void**)&data, size, flags),
		              BF_STATUS_MEM_ALLOC_FAILED);
		break;
	}
#endif
	default: BF_FAIL("Valid bfMalloc() space", BF_STATUS_INVALID_SPACE);
	}
	//return data;
	*ptr = data;
	return BF_STATUS_SUCCESS;
}
BFstatus bfFree(void* ptr, BFspace space) {
	BF_ASSERT(ptr, BF_STATUS_INVALID_POINTER);
	if( space == BF_SPACE_AUTO ) {
		bfGetSpace(ptr, &space);
	}
	switch( space ) {
	case BF_SPACE_SYSTEM:       ::free(ptr); break;
#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
	case BF_SPACE_CUDA:         cudaFree(ptr); break;
	case BF_SPACE_CUDA_HOST:    cudaFreeHost(ptr); break;
	case BF_SPACE_CUDA_MANAGED: cudaFree(ptr); break;
#endif
	default: BF_FAIL("Valid bfFree() space", BF_STATUS_INVALID_ARGUMENT);
	}
	return BF_STATUS_SUCCESS;
}
BFstatus bfMemcpy(void*       dst,
                  BFspace     dst_space,
                  const void* src,
                  BFspace     src_space,
                  BFsize      count) {
	if( count ) {
		BF_ASSERT(dst, BF_STATUS_INVALID_POINTER);
		BF_ASSERT(src, BF_STATUS_INVALID_POINTER);
#if !defined BF_CUDA_ENABLED || !BF_CUDA_ENABLED
		::memcpy(dst, src, count);
#else
		// Note: Explicitly dispatching to ::memcpy was found to be much faster
		//         than using cudaMemcpyDefault.
		if( src_space == BF_SPACE_AUTO ) bfGetSpace(src, &src_space);
		if( dst_space == BF_SPACE_AUTO ) bfGetSpace(dst, &dst_space);
		cudaMemcpyKind kind = cudaMemcpyDefault;
		switch( src_space ) {
		case BF_SPACE_CUDA_HOST: // fall-through
		case BF_SPACE_SYSTEM: {
			switch( dst_space ) {
			case BF_SPACE_CUDA_HOST: // fall-through
			case BF_SPACE_SYSTEM: ::memcpy(dst, src, count); return BF_STATUS_SUCCESS;
			case BF_SPACE_CUDA: kind = cudaMemcpyHostToDevice; break;
			// TODO: BF_SPACE_CUDA_MANAGED
			default: BF_FAIL("Valid bfMemcpy dst space", BF_STATUS_INVALID_ARGUMENT);
			}
			break;
		}
		case BF_SPACE_CUDA: {
			switch( dst_space ) {
			case BF_SPACE_CUDA_HOST: // fall-through
			case BF_SPACE_SYSTEM: kind = cudaMemcpyDeviceToHost; break;
			case BF_SPACE_CUDA:   kind = cudaMemcpyDeviceToDevice; break;
			// TODO: BF_SPACE_CUDA_MANAGED
			default: BF_FAIL("Valid bfMemcpy dst space", BF_STATUS_INVALID_ARGUMENT);
			}
			break;
		}
		default: BF_FAIL("Valid bfMemcpy src space", BF_STATUS_INVALID_ARGUMENT);
		}
		BF_TRACE_STREAM(g_cuda_stream);
		BF_CHECK_CUDA(cudaMemcpyAsync(dst, src, count, kind, g_cuda_stream),
		              BF_STATUS_MEM_OP_FAILED);
#endif
	}
	return BF_STATUS_SUCCESS;
}
void memcpy2D(void*       dst,
              BFsize      dst_stride,
              const void* src,
              BFsize      src_stride,
              BFsize      width,
              BFsize      height) {
	//std::cout << "memcpy2D dst: " << dst << ", " << dst_stride << std::endl;
	//std::cout << "memcpy2D src: " << src << ", " << src_stride << std::endl;
	//std::cout << "memcpy2D shp: " << width << ", " << height << std::endl;
	for( BFsize row=0; row<height; ++row ) {
		::memcpy((char*)dst + row*dst_stride,
		         (char*)src + row*src_stride,
		         width);
	}
}
BFstatus bfMemcpy2D(void*       dst,
                    BFsize      dst_stride,
                    BFspace     dst_space,
                    const void* src,
                    BFsize      src_stride,
                    BFspace     src_space,
                    BFsize      width,    // bytes
                    BFsize      height) { // rows
	if( width*height ) {
		BF_ASSERT(dst, BF_STATUS_INVALID_POINTER);
		BF_ASSERT(src, BF_STATUS_INVALID_POINTER);
#if !defined BF_CUDA_ENABLED || !BF_CUDA_ENABLED
		memcpy2D(dst, dst_stride, src, src_stride, width, height);
#else
		// Note: Explicitly dispatching to ::memcpy was found to be much faster
		//         than using cudaMemcpyDefault.
		if( src_space == BF_SPACE_AUTO ) bfGetSpace(src, &src_space);
		if( dst_space == BF_SPACE_AUTO ) bfGetSpace(dst, &dst_space);
		cudaMemcpyKind kind = cudaMemcpyDefault;
		switch( src_space ) {
		case BF_SPACE_CUDA_HOST: // fall-through
		case BF_SPACE_SYSTEM: {
			switch( dst_space ) {
			case BF_SPACE_CUDA_HOST: // fall-through
			case BF_SPACE_SYSTEM: memcpy2D(dst, dst_stride, src, src_stride, width, height); return BF_STATUS_SUCCESS;
			case BF_SPACE_CUDA: kind = cudaMemcpyHostToDevice; break;
			// TODO: Is this the right thing to do?
			case BF_SPACE_CUDA_MANAGED: kind = cudaMemcpyDefault; break;
			default: BF_FAIL("Valid bfMemcpy2D dst space", BF_STATUS_INVALID_ARGUMENT);
			}
			break;
		}
		case BF_SPACE_CUDA: {
			switch( dst_space ) {
			case BF_SPACE_CUDA_HOST: // fall-through
			case BF_SPACE_SYSTEM: kind = cudaMemcpyDeviceToHost; break;
			case BF_SPACE_CUDA:   kind = cudaMemcpyDeviceToDevice; break;
			// TODO: Is this the right thing to do?
			case BF_SPACE_CUDA_MANAGED: kind = cudaMemcpyDefault; break;
			default: BF_FAIL("Valid bfMemcpy2D dst space", BF_STATUS_INVALID_ARGUMENT);
			}
			break;
		}
		default: BF_FAIL("Valid bfMemcpy2D src space", BF_STATUS_INVALID_ARGUMENT);
		}
		BF_TRACE_STREAM(g_cuda_stream);
		BF_CHECK_CUDA(cudaMemcpy2DAsync(dst, dst_stride,
		                                src, src_stride,
		                                width, height,
		                                kind, g_cuda_stream),
		              BF_STATUS_MEM_OP_FAILED);
#endif
	}
	return BF_STATUS_SUCCESS;
}
BFstatus bfMemset(void*   ptr,
                  BFspace space,
                  int     value,
                  BFsize  count) {
	BF_ASSERT(ptr, BF_STATUS_INVALID_POINTER);
	if( count ) {
		if( space == BF_SPACE_AUTO ) {
			// TODO: Check status here
			bfGetSpace(ptr, &space);
		}
		switch( space ) {
		case BF_SPACE_SYSTEM:       ::memset(ptr, value, count); break;
#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
		case BF_SPACE_CUDA_HOST:    ::memset(ptr, value, count); break;
		case BF_SPACE_CUDA: // Fall-through
		case BF_SPACE_CUDA_MANAGED: {
			BF_TRACE_STREAM(g_cuda_stream);
			BF_CHECK_CUDA(cudaMemsetAsync(ptr, value, count, g_cuda_stream),
			              BF_STATUS_MEM_OP_FAILED);
			break;
		}
#endif
		default: BF_FAIL("Valid bfMemset space", BF_STATUS_INVALID_ARGUMENT);
		}
	}
	return BF_STATUS_SUCCESS;
}
void memset2D(void*  ptr,
              BFsize stride,
              int    value,
              BFsize width,
              BFsize height) {
	for( BFsize row=0; row<height; ++row ) {
		::memset((char*)ptr + row*stride, value, width);
	}
}
BFstatus bfMemset2D(void*   ptr,
                    BFsize  stride,
                    BFspace space,
                    int     value,
                    BFsize  width,    // bytes
                    BFsize  height) { // rows
	BF_ASSERT(ptr, BF_STATUS_INVALID_POINTER);
	if( width*height ) {
		if( space == BF_SPACE_AUTO ) {
			bfGetSpace(ptr, &space);
		}
		switch( space ) {
		case BF_SPACE_SYSTEM:       memset2D(ptr, stride, value, width, height); break;
#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
		case BF_SPACE_CUDA_HOST:    memset2D(ptr, stride, value, width, height); break;
		case BF_SPACE_CUDA: // Fall-through
		case BF_SPACE_CUDA_MANAGED: {
			BF_TRACE_STREAM(g_cuda_stream);
			BF_CHECK_CUDA(cudaMemset2DAsync(ptr, stride, value, width, height, g_cuda_stream),
			              BF_STATUS_MEM_OP_FAILED);
			break;
		}
#endif
		default: BF_FAIL("Valid bfMemset2D space", BF_STATUS_INVALID_ARGUMENT);
		}
	}
	return BF_STATUS_SUCCESS;
}
BFsize bfGetAlignment() {
	return BF_ALIGNMENT;
}
