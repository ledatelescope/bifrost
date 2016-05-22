/*
 *  Copyright 2015 Ben Barsdell
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
	void check_error(cudaError_t ret) const {
		if( ret != cudaSuccess ) {
			throw std::runtime_error(cudaGetErrorString(ret));
		}
	}
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

} // namespace cuda
