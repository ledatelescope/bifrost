/*
 *  Copyright 2016 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <bifrost/memory.h>

#if BF_CUDA_ENABLED
  #include "cuda/stream.hpp"
  #include <cuda_runtime_api.h>
#endif

#include <cstdlib> // For posix_memalign
#include <cstring> // For memcpy
#include <iostream>

#define BF_IS_POW2(x) (x) && !((x) & ((x) - 1))
static_assert(BF_IS_POW2(BF_ALIGNMENT), "BF_ALIGNMENT must be a power of 2");
#undef BF_IS_POW2
//static_assert(BF_ALIGNMENT >= 8,        "BF_ALIGNMENT must be >= 8");

// TODO: This is duplicated in bfring.cpp; move it to a shared header
#if defined(BF_DEBUG) && BF_DEBUG
#define BF_RETURN_ERROR(err) do { \
		std::cerr << __FILE__ << ":" << __LINE__ \
		          << "error " << err << ": " \
		          << bfGetStatusString(err) << std::endl; \
			return (err); \
	} while(0)
#else
#define BF_RETURN_ERROR(err) return (err)
#endif // BF_DEBUG

#define BF_ASSERT(pred, err) do { \
		if( !(pred) ) { \
			BF_RETURN_ERROR(err); \
		} \
	} while(0)

// TODO: Change this to return status as per the library convention
BFspace bfGetSpace(const void* ptr, BFstatus* status) {
	if( status ) *status = BF_STATUS_SUCCESS;
#if !defined BF_CUDA_ENABLED || !BF_CUDA_ENABLED
	return BF_SPACE_SYSTEM;
#else
	cudaPointerAttributes ptr_attrs;
	cudaPointerGetAttributes(&ptr_attrs, ptr);
	if( ptr_attrs.isManaged ) {
		return BF_SPACE_CUDA_MANAGED;
	}
	switch( ptr_attrs.memoryType ) {
	case cudaMemoryTypeHost:   return BF_SPACE_SYSTEM;
	case cudaMemoryTypeDevice: return BF_SPACE_CUDA;
	default: {
		if( status ) *status = BF_STATUS_INVALID_POINTER;
		return BF_SPACE_AUTO; // TODO: Any better option than this?
	}
	}
#endif
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
		cudaError_t err = cudaMalloc((void**)&data, size);
		BF_ASSERT(err == cudaSuccess, BF_STATUS_MEM_ALLOC_FAILED);
		break;
	}
	case BF_SPACE_CUDA_HOST: {
		unsigned flags = cudaHostAllocDefault;
		cudaError_t err = cudaHostAlloc((void**)&data, size, flags);
		BF_ASSERT(err == cudaSuccess, BF_STATUS_MEM_ALLOC_FAILED);
		break;
	}
	case BF_SPACE_CUDA_MANAGED: {
		unsigned flags = cudaMemAttachGlobal;
		cudaError_t err = cudaMallocManaged((void**)&data, size, flags);
		BF_ASSERT(err == cudaSuccess, BF_STATUS_MEM_ALLOC_FAILED);
		break;
	}
#endif
	default: BF_ASSERT(false, BF_STATUS_INVALID_ARGUMENT);
	}
	//return data;
	*ptr = data;
	return BF_STATUS_SUCCESS;
}
BFstatus bfFree(void* ptr, BFspace space) {
	BF_ASSERT(ptr, BF_STATUS_INVALID_POINTER);
	if( space == BF_SPACE_AUTO ) {
		space = bfGetSpace(ptr, 0);
	}
	switch( space ) {
	case BF_SPACE_SYSTEM:       ::free(ptr); break;
#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
	case BF_SPACE_CUDA:         cudaFree(ptr); break;
	case BF_SPACE_CUDA_HOST:    cudaFreeHost(ptr); break;
	case BF_SPACE_CUDA_MANAGED: cudaFree(ptr); break;
#endif
	default: BF_ASSERT(false, BF_STATUS_INVALID_ARGUMENT);
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
		if( src_space == BF_SPACE_AUTO ) src_space = bfGetSpace(src, 0);
		if( dst_space == BF_SPACE_AUTO ) dst_space = bfGetSpace(dst, 0);
		cudaMemcpyKind kind = cudaMemcpyDefault;
		switch( src_space ) {
		case BF_SPACE_SYSTEM: {
			switch( dst_space ) {
			case BF_SPACE_CUDA_HOST: // fall-through
			case BF_SPACE_SYSTEM: ::memcpy(dst, src, count); return BF_STATUS_SUCCESS;
			case BF_SPACE_CUDA: kind = cudaMemcpyHostToDevice; break;
			// TODO: BF_SPACE_CUDA_MANAGED
			default: return BF_STATUS_INVALID_ARGUMENT;
			}
			break;
		}
		case BF_SPACE_CUDA: {
			switch( dst_space ) {
			case BF_SPACE_CUDA_HOST: // fall-through
			case BF_SPACE_SYSTEM: kind = cudaMemcpyDeviceToHost; break;
			case BF_SPACE_CUDA:   kind = cudaMemcpyDeviceToDevice; break;
			// TODO: BF_SPACE_CUDA_MANAGED
			default: return BF_STATUS_INVALID_ARGUMENT;
			}
			break;
		}
		default: return BF_STATUS_INVALID_ARGUMENT;
		}
		cuda::scoped_stream s;
		if( cudaMemcpyAsync(dst, src, count, kind, s) != cudaSuccess ) {
			return BF_STATUS_MEM_OP_FAILED;
		}
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
                    BFsize      width,
                    BFsize      height) {
	if( width*height ) {
		BF_ASSERT(dst, BF_STATUS_INVALID_POINTER);
		BF_ASSERT(src, BF_STATUS_INVALID_POINTER);
#if !defined BF_CUDA_ENABLED || !BF_CUDA_ENABLED
		memcpy2D(dst, dst_stride, src, src_stride, width, height);
#else
		// Note: Explicitly dispatching to ::memcpy was found to be much faster
		//         than using cudaMemcpyDefault.
		if( src_space == BF_SPACE_AUTO ) src_space = bfGetSpace(src, 0);
		if( dst_space == BF_SPACE_AUTO ) dst_space = bfGetSpace(dst, 0);
		cudaMemcpyKind kind = cudaMemcpyDefault;
		switch( src_space ) {
		case BF_SPACE_SYSTEM: {
			switch( dst_space ) {
			case BF_SPACE_CUDA_HOST: // fall-through
			case BF_SPACE_SYSTEM: memcpy2D(dst, dst_stride, src, src_stride, width, height); return BF_STATUS_SUCCESS;
			case BF_SPACE_CUDA: kind = cudaMemcpyHostToDevice; break;
			// TODO: Is this the right thing to do?
			case BF_SPACE_CUDA_MANAGED: kind = cudaMemcpyDefault; break;
			default: return BF_STATUS_INVALID_ARGUMENT;
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
			default: return BF_STATUS_INVALID_ARGUMENT;
			}
			break;
		}
		default: return BF_STATUS_INVALID_ARGUMENT;
		}
		cuda::scoped_stream s;
		if( cudaMemcpy2DAsync(dst, dst_stride,
		                      src, src_stride,
		                      width, height,
		                      kind, s) != cudaSuccess ) {
			return BF_STATUS_MEM_OP_FAILED;
		}
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
			space = bfGetSpace(ptr, 0);
		}
		switch( space ) {
		case BF_SPACE_SYSTEM:       ::memset(ptr, value, count); break;
#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
		case BF_SPACE_CUDA_HOST:    ::memset(ptr, value, count); break;
		case BF_SPACE_CUDA:
		case BF_SPACE_CUDA_MANAGED: {
			cuda::scoped_stream s;
			cudaMemsetAsync(ptr, value, count, s);
			break;
		}
#endif
		default: return BF_STATUS_INVALID_ARGUMENT;
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
                    BFsize  width,
                    BFsize  height) {
	BF_ASSERT(ptr, BF_STATUS_INVALID_POINTER);
	if( width*height ) {
		if( space == BF_SPACE_AUTO ) {
			space = bfGetSpace(ptr, 0);
		}
		switch( space ) {
		case BF_SPACE_SYSTEM:       memset2D(ptr, stride, value, width, height); break;
#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
		case BF_SPACE_CUDA_HOST:    memset2D(ptr, stride, value, width, height); break;
		case BF_SPACE_CUDA:
		case BF_SPACE_CUDA_MANAGED: {
			cuda::scoped_stream s;
			cudaMemset2DAsync(ptr, stride, value, width, height, s);
			break;
		}
#endif
		default: return BF_STATUS_INVALID_ARGUMENT;
		}
	}
	return BF_STATUS_SUCCESS;
}
BFsize bfGetAlignment() {
	return BF_ALIGNMENT;
}
