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
#include "assert.hpp"

#include <string>
#include <vector>
#include <stdexcept>
#include <map>

#if BF_CUDA_ENABLED

//#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <type_traits>

extern thread_local cudaStream_t g_cuda_stream;
// TODO: BFstatus bfSetStream(void const* stream) { cuda_stream = *(cudaStream_t*)stream; }

#if THRUST_VERSION >= 100800 // WAR for old Thrust version (e.g., on TK1)
 // TODO: Also need thrust::cuda::par(allocator).on(stream)
 #define thrust_cuda_par_on(stream) thrust::cuda::par.on(stream)
#else
 // TODO: Also need thrust::cuda::par(allocator, stream)
 #define thrust_cuda_par_on(stream) thrust::cuda::par(stream)
#endif

inline int get_cuda_device_cc() {
	int device;
	cudaGetDevice(&device);
	int cc_major;
	cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device);
	int cc_minor;
	cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device);
	return cc_major*10 + cc_minor;
}

#define BF_CHECK_CUDA_EXCEPTION(call, err) \
	do { \
		cudaError_t cuda_ret = call; \
		if( cuda_ret != cudaSuccess ) { \
			BF_DEBUG_PRINT(cudaGetErrorString(cuda_ret)); \
		} \
		/*BF_ASSERT(cuda_ret == cudaSuccess, err);*/ \
		BF_ASSERT_EXCEPTION(cuda_ret == cudaSuccess, err); \
	} while(0)

#define BF_CHECK_CUDA(call, err) \
	do { \
		cudaError_t cuda_ret = call; \
		if( cuda_ret != cudaSuccess ) { \
			BF_DEBUG_PRINT(cudaGetErrorString(cuda_ret)); \
		} \
		BF_ASSERT(cuda_ret == cudaSuccess, err); \
	} while(0)

class CUDAKernel {
	CUmodule                  _module;
	CUfunction                _kernel;
	std::string               _func_name;
	std::string               _ptx;
	std::vector<CUjit_option> _opts;
	
	inline void cuda_safe_call(CUresult res) {
		if( res != CUDA_SUCCESS ) {
			const char* msg;
			cuGetErrorName(res, &msg);
			throw std::runtime_error(msg);
		}
	}
	inline void create_module(void** optvals=0) {
		cuda_safe_call(cuModuleLoadDataEx(&_module, _ptx.c_str(),
		                                  _opts.size(), &_opts[0], optvals));
		cuda_safe_call(cuModuleGetFunction(&_kernel, _module,
		                                   _func_name.c_str()));
	}
	inline void destroy_module() {
		if( _module ) {
			cuModuleUnload(_module);
		}
	}
public:
	inline CUDAKernel() : _module(0), _kernel(0) {}
	inline CUDAKernel(const CUDAKernel& other) : _module(0), _kernel(0) {
		if( other._module ) {
			_func_name = other._func_name;
			_ptx       = other._ptx;
			_opts      = other._opts;
			this->create_module();
		}
	}
	inline CUDAKernel(const char*   func_name,
	                  const char*   ptx,
	                  unsigned int  nopts=0,
	                  CUjit_option* opts=0,
	                  void**        optvals=0) {
		_func_name = func_name;
		_ptx = ptx;
		_opts.assign(opts, opts + nopts);
		this->create_module(optvals);
	}
	inline CUDAKernel& set(const char*   func_name,
	                       const char*   ptx,
	                       unsigned int  nopts=0,
	                       CUjit_option* opts=0,
	                       void**        optvals=0) {
		this->destroy_module();
		_func_name = func_name;
		_ptx = ptx;
		_opts.assign(opts, opts + nopts);
		this->create_module(optvals);
		return *this;
	}
	inline void swap(CUDAKernel& other) {
		std::swap(_func_name, other._func_name);
		std::swap(_ptx, other._ptx);
		std::swap(_opts, other._opts);
		std::swap(_module, other._module);
		std::swap(_kernel, other._kernel);
	}
	inline CUDAKernel& operator=(const CUDAKernel& other) {
		CUDAKernel tmp(other);
		this->swap(tmp);
		return *this;
	}
	inline ~CUDAKernel() {
		this->destroy_module();
	}
	inline operator CUfunction() const { return _kernel; }
	
	inline CUresult launch(dim3 grid, dim3 block,
	                       unsigned int smem, CUstream stream,
	                       std::vector<void*> arg_ptrs) {
	                       //void* arg_ptrs[]) {
		// Note: This returns "INVALID_ARGUMENT" if 'args' do not match what is
		//         expected (e.g., too few args, wrong types)
		return cuLaunchKernel(_kernel,
		                      grid.x, grid.y, grid.z,
		                      block.x, block.y, block.z,
		                      smem, stream,
		                      &arg_ptrs[0], NULL);
	}
#if __cplusplus >= 201103L
	template<typename... Args>
	inline CUresult launch(dim3 grid, dim3 block,
	                   unsigned int smem, CUstream stream,
	                   Args... args) {
		return this->launch(grid, block, smem, stream, {(void*)&args...});
	}
#endif
	
};

#else // BF_CUDA_ENABLED

#define __host__
#define __device__

#endif // BF_CUDA_ENABLED
