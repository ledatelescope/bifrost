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

/*! \file stream.hpp
 *  \brief Feature-complete RAII wrapper for CUDA stream objects
 */

/*
  Note: According to Steven Jones's GTC2015 talk[1], streams are very cheap
          to create and destroy, so this can readily be done on-the-fly.
          [1] http://on-demand.gputechconf.com/gtc/2015/presentation/S5530-Stephen-Jones.pdf
              http://on-demand.gputechconf.com/gtc/2015/video/S5530.html

Example showing different ways to run two memcopies in parallel
---------------------------------------------------------------
void async_memcpy(void* dst, const void* src, size_t size) {
  cuda::stream stream;
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
}
void device_async_memcpy(void* dst, const void* src, size_t size) {
  CUDAScopedStream stream;
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
}
// C++11 only
cuda::stream async_memcpy_cxx11(void* dst, const void* src, size_t size) {
  cuda::stream stream(0, cudaStreamNonBlocking);
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
  return stream;
}
int main() {
  // ...
  async_memcpy(dst1, src2, size);
  async_memcpy(dst2, src2, size);
  cudaDeviceSynchronize();
  // Or
  std::thread t1( device_async_memcpy(dst, src, size) );
  std::thread t2( device_async_memcpy(dst, src, size) );
  t1.join();
  t2.join();
  // Or (C++11 only)
  cuda::stream s1(async_memcpy_cxx11(dst1, src2, size));
  cuda::stream s2;
  s2 = async_memcpy_cxx11(dst2, src2, size);
  s1.synchronize();
  s2.synchronize();
  // ...
}
 */

#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>

namespace cuda {

inline void check_error(cudaError_t ret) {
	if( ret != cudaSuccess ) {
		throw std::runtime_error(cudaGetErrorString(ret));
	}
}

class stream {
	cudaStream_t _obj;
	// Not copy-assignable
#if __cplusplus >= 201103L
	stream(const cuda::stream& other) = delete;
	stream& operator=(const cuda::stream& other) = delete;
#else
	stream(const cuda::stream& other);
	stream& operator=(const cuda::stream& other);
#endif
	void destroy() { if( _obj ) { cudaStreamDestroy(_obj); _obj = 0; } }
public:
#if __cplusplus >= 201103L
	// Move semantics
	inline stream(cuda::stream&& other) : _obj(0) { this->swap(other); }
	inline cuda::stream& operator=(cuda::stream&& other) {
		this->destroy();
		this->swap(other);
		return *this;
	}
#endif
	inline explicit stream(int      priority=0,
	                       unsigned flags=cudaStreamDefault) : _obj(0) {
		if( priority > 0 ) {
			int least_priority;
			int greatest_priority;
			cudaDeviceGetStreamPriorityRange(&least_priority,
			                                 &greatest_priority);
			check_error( cudaStreamCreateWithPriority(&_obj,
			                                          flags,
			                                          greatest_priority) );
		}
		else {
			check_error( cudaStreamCreateWithFlags(&_obj, flags) );
		}
	}
	inline ~stream() { this->destroy(); }
	inline void swap(cuda::stream& other) { std::swap(_obj, other._obj); }
	inline int priority() const {
		int val;
		check_error( cudaStreamGetPriority(_obj, &val) );
		return val;
	}
	inline unsigned flags() const {
		unsigned val;
		check_error( cudaStreamGetFlags(_obj, &val) );
		return val;
	}
	inline bool query() const {
		cudaError_t ret = cudaStreamQuery(_obj);
		if( ret == cudaErrorNotReady ) {
			return false;
		}
		else {
			check_error(ret);
			return true;
		}
	}
	inline void synchronize() const {
		cudaStreamSynchronize(_obj);
		check_error( cudaGetLastError() );
	}
	inline void wait(cudaEvent_t event, unsigned flags=0) const {
		check_error( cudaStreamWaitEvent(_obj, event, flags) );
	}
	inline void addCallback(cudaStreamCallback_t callback,
	                 void* userData=0, unsigned flags=0) {
		check_error( cudaStreamAddCallback(_obj, callback, userData, flags) );
	}
	inline void attachMemAsync(void* devPtr, size_t length, unsigned flags) {
		check_error( cudaStreamAttachMemAsync(_obj, devPtr, length, flags) );
	}
	inline operator const cudaStream_t&() const { return _obj; }
};
// This version automatically calls synchronize() before destruction
class scoped_stream : public cuda::stream {
	typedef cuda::stream super_type;
public:
	inline explicit scoped_stream(int      priority=0,
	                              unsigned flags=cudaStreamNonBlocking)
		: super_type(priority, flags) {}
	inline ~scoped_stream() { this->synchronize(); }
};
// This version automatically syncs with a parent stream on construct/destruct
class child_stream : public cuda::stream {
	typedef cuda::stream super_type;
	cudaStream_t _parent;
	void sync_streams(cudaStream_t dependent, cudaStream_t dependee) {
		// Record event in dependee and make dependent wait for it
		cudaEvent_t event;
		check_error(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		check_error(cudaEventRecord(event, dependee));
		check_error(cudaStreamWaitEvent(dependent, event, 0));
		check_error(cudaEventDestroy(event));
	}
public:
	inline explicit child_stream(cudaStream_t parent,
	                             int          priority=0,
	                             unsigned     flags=cudaStreamNonBlocking)
		: super_type(priority, flags), _parent(parent) {
		sync_streams(*this, _parent);
	}
	inline ~child_stream() {
		sync_streams(_parent, *this);
	}
};

} // namespace cuda
