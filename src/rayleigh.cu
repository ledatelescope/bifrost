/*
 * Copyright (c) 2022, The Bifrost Authors. All rights reserved.
 * Copyright (c) 2022, The University of New Mexico. All rights reserved.
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

#include <bifrost/fir.h>
#include "assert.hpp"
#include "utils.hpp"
#include "workspace.hpp"
#include "cuda.hpp"
#include "trace.hpp"
#include "Complex.hpp"

//#include <limits>

#include <math_constants.h> // For CUDART_NAN_F
#include <thrust/device_vector.h>
#include <curand.h>

#include <vector>
#include <map>
#include <string>

// HACK TESTING
#include <iostream>
using std::cout;
using std::endl;

#define BF_POOL_SIZE 268435456

#define BF_CHECK_CURAND_EXCEPTION(call, err) \
	do { \
		cudaError_t cuda_ret = call; \
		if( cuda_ret != CURAND_STATUS_SUCCESS ) { \
			BF_DEBUG_PRINT(cudaGetErrorString(cuda_ret)); \
		} \
		/*BF_ASSERT(cuda_ret == cudaSuccess, err);*/ \
		BF_ASSERT_EXCEPTION(cuda_ret == cudaSuccess, err); \
	} while(0)

#define BF_CHECK_CURAND(call, err) \
	do { \
		cudaError_t cuda_ret = call; \
		if( cuda_ret != CURAND_STATUS_SUCCESS ) { \
			BF_DEBUG_PRINT(cudaGetErrorString(cuda_ret)); \
		} \
		BF_ASSERT(cuda_ret == CURAND_STATUS_SUCCES, err); \
	} while(0)

template<typename InType>
__global__ void flagger_kernel(unsigned int               ntime, 
                               unsigned int               nantpol,
                               float                      alpha,
                               unsigned int               clip_sigmas,
                               float                      max_flag_frac,
                               float*                     state,
                               const float*               pool,
                               unsigned int*              flags,
                               const InType* __restrict__ d_in,
                               InType* __restrict__       d_out) {
	int a = threadIdx.x + blockIdx.x*blockDim.x;
	
  int r = a;
  if( r > BF_POOL_SIZE ) r %= BF_POOL_SIZE;
  
	int c, t, t0;
  int count, bad_count;
  float mean;
	float power;
	if( a < nantpol ) {
    mean = 0.0;
    count = 0.0;
    
    
		for(t=0; t<ntime; t++) {
      power  = d_in[t*nantpol*2 + a*2 + 0]*d_in[t*nantpol*2 + a*2 + 0];
      power += d_in[t*nantpol*2 + a*2 + 1]*d_in[t*nantpol*2 + a*2 + 1];
      
      mean += power;
      count += 1;
      
      if( power >= clip_sigmas*sqrt(4/D_PI -1)*state[a]) ) {
        d_out[t*nantpol*2 + a*2 + 0] = pool[r++] * sqrt(2/D_PI)*state[a];
        if( r > BF_POOL_SIZE ) r = 0;
        d_out[t*nantpol*2 + a*2 + 1] = pool[r++] * sqrt(2/D_PI)*state[a];
        if( r > BF_POOL_SIZE ) r = 0;
        bad_count += 1;
      } else {
        d_out[t*nantpol*2 + a*2 + 0] = d_in[t*nantpol*2 + a*2 + 0];
        d_out[t*nantpol*2 + a*2 + 1] = d_in[t*nantpol*2 + a*2 + 1];
			}
		}
    
    mean /= count;
    if( bad_count < (count*max_flag_frac)) {
      state[a] = alpha*mean + (1-alpha)*state[a];
    } else {
      atomicAdd(flags, 1);
    }
	}
}

template<typename InType>
inline void launch_flagger_kernel(unsigned int ntime, 
                                  unsigned int nantpol,
                                  float        alpha,
                                  unsigned int clip_sigmas,
                                  float        max_flag_frac,
                                  float*       state,
                                  float*       pool,
                                  usigned int* flags,
                                  InType*      d_in,
                                  InType*      d_out,
                                  cudaStream_t stream=0) {
	//cout << "LAUNCH for " << nelement << endl;
	dim3 block(std::min(256u, nantpol), 256u/std::min(256u, nantpol));
	int first = std::min((nantpol-1)/block.x+1, 65535u);
	dim3 grid(first, 1u, 1u);

	cout << "  Block size is " << block.x << " by " << block.y << endl;
	cout << "  Grid  size is " << grid.x << " by " << grid.y << " by " << grid.z << endl;
	cout << "  Maximum size is " << block.y*grid.y*grid.z << endl;
	if( block.y*grid.y*grid.z >= ntime ) {
		cout << "  -> Valid" << endl;
	}

	
	void* args[] = {&ntime, 
	                &nantpol,
	                &alpha,
                  &clip_sigmas,
	                &max_flag_frac
	                &state,
                  &pool,
                  &flags,
	                &d_in,
	                &d_out};
	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)flagger_kernel<InType>,
	                                         grid, block,
	                                         &args[0], 0, stream),
	                        BF_STATUS_INTERNAL_ERROR);
}

class BFrayleigh_impl {
	typedef int          IType;
	typedef unsigned int UType;
	typedef double       FType;
public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
	typedef float  DType;
private:
	UType     _nantpol;
  float     _alpha;
  UType     _clip_sigmas;
  float     _max_flag_frac;
  float*    _state = NULL;
  float*    _pool = NULL;
	IType     _plan_stride;
	Workspace _plan_storage;
	// TODO: Use something other than Thrust
	thrust::device_vector<char> _dv_plan_storage;
	cudaStream_t _stream;
public:
	BFrayleigh_impl() : _stream(g_cuda_stream) {}
	inline UType nantpol()  const { return _nantpol; }
	void init(UType nantpol, 
            float alpha,
	          UType clip_sigmas,
            float max_flag_frac) {
		BF_TRACE();
		_nantpol       = nantpol;
    _alpha         = alpha;
    _clip_sigmas   = clip_sigmas;
    _max_flag_frac = max_flag_frac;
		
		_state = NULL;
	}
	bool init_plan_storage(void* storage_ptr, BFsize* storage_size) {
		BF_TRACE();
		BF_TRACE_STREAM(_stream);
		enum {
			ALIGNMENT_BYTES = 512,
			ALIGNMENT_ELMTS = ALIGNMENT_BYTES / sizeof(float)
		};
		Workspace workspace(ALIGNMENT_BYTES);
		_plan_stride = round_up(_nantpol, ALIGNMENT_ELMTS);
		workspace.reserve(_nantpol+1, &_state);
    workspace.reserve((_nantpol+1)*BF_POOL_SIZE, &_pool);
		if( storage_size ) {
			if( !storage_ptr ) {
				// Return required storage size
				*storage_size = workspace.size();
				return false;
			} else {
				BF_ASSERT_EXCEPTION(*storage_size >= workspace.size(),
				                    BF_STATUS_INSUFFICIENT_STORAGE);
			}
		} else {
			// Auto-allocate storage
			BF_ASSERT_EXCEPTION(!storage_ptr, BF_STATUS_INVALID_ARGUMENT);
			_dv_plan_storage.resize(workspace.size());
			storage_ptr = thrust::raw_pointer_cast(&_dv_plan_storage[0]);
		}
		workspace.commit(storage_ptr);
		
		this->reset_state();
		return true;
	}
	void reset_state() {
		BF_ASSERT_EXCEPTION(_state != NULL,  BF_STATUS_INVALID_STATE);
		
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
		
		// Reset the state
		BF_CHECK_CUDA_EXCEPTION( cudaMemsetAsync(_state,
		                                         0,
		                                         sizeof(float)*_nantpol,
		                                         _stream),
		                         BF_STATUS_MEM_OP_FAILED );
		BF_CHECK_CUDA_EXCEPTION( cudaStreamSynchronize(_stream),
		                         BF_STATUS_DEVICE_ERROR );
		
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
    
    curandGenerator_t gen;
    BF_CHECK_CURAND_EXCEPTION(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT),
                              BF_STATUS_DEVICE_ERROR);
    BF_CHECK_CURAND_EXCEPTION(curandSetPseudoRandomGeneratorSeed(gen, 98105102114111115116ULL),
                              BF_STATUS_DEVICE_ERROR);
    BF_CHECK_CURAND_EXCEPTION(curandGenerateNormal(gen, _pool, BF_POOL_SIZE, 0.0, 1.0),
                              BF_STATUS_DEVICE_ERROR);
    BF_CHECK_CURAND_EXCEPTION(curandDestroyGenerator(gen), BF_STATUS_DEVICE_ERROR);
	}
	void execute(BFarray const* in,
	             BFarray const* out,
               BFsize*        flags) {
		BF_TRACE();
		BF_TRACE_STREAM(_stream);
		BF_ASSERT_EXCEPTION(_state != NULL, BF_STATUS_INVALID_STATE);
		BF_ASSERT_EXCEPTION(out->dtype == BF_DTYPE_CF32 || \
		                    out->dtype == BF_DTYPE_CF64,     BF_STATUS_UNSUPPORTED_DTYPE);
		
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
		
#define LAUNCH_FLAGGER_KERNEL(IterType) \
		launch_flagger_kernel(in->shape[0], _nantpol, \
                          _alpha, _clip_sigmas, _max_flag_frac, \
		                      _state, _pool, flags, \
		                      (IterType)in->data, (IterType)out->data, \
		                      _stream)
		
    *flags = 0;
		switch( in->dtype ) {
			case BF_DTYPE_CI8:  LAUNCH_FLAGGER_KERNEL(int8_t*);  break;
			case BF_DTYPE_CI16: LAUNCH_FLAGGER_KERNEL(int16_t*); break;
			case BF_DTYPE_CI32: LAUNCH_FLAGGER_KERNEL(int32_t*); break;
			case BF_DTYPE_CI64: LAUNCH_FLAGGER_KERNEL(int64_t*); break;
			case BF_DTYPE_CF32: LAUNCH_FLAGGER_KERNEL(float*);   break;
			case BF_DTYPE_CF64: LAUNCH_FLAGGER_KERNEL(double*);  break;
			default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
		}
#undef LAUNCH_FLAGGER_KERNEL
	}
	void set_stream(cudaStream_t stream) {
		_stream = stream;
	}
};

BFstatus bfRayleighCreate(BFfir* plan_ptr) {
	BF_TRACE();
	BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*plan_ptr = new BFrayleigh_impl(),
	                   *plan_ptr = 0);
}

BFstatus bfRayleighInit(BFrayleigh plan,
                        BFsize     nantpols,
                        float      alpha,
                        BFsize     clip_sigmas,
                        float      max_flag_frac,
                        BFspace    space,
                        void*      plan_storage,
                        BFsize*    plan_storage_size) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(space_accessible_from(space, BF_SPACE_CUDA),
	          BF_STATUS_UNSUPPORTED_SPACE);
  
  BF_ASSERT((alpha > 0) && (alpha <= 1), BF_STATUS_INVALID_ARGUMENT);
	
	BF_TRY(plan->init(nantpols, alpha, clip_sigmas, max_flag_frac));
	BF_TRY_RETURN(plan->init_plan_storage(plan_storage, plan_storage_size));
}
BFstatus bfRayleighSetStream(BFrayleigh  plan,
                             void const* stream) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}
BFstatus bfRayleighResetState(BFrayleigh plan) {
	BF_TRY_RETURN(plan->reset_state());
}
BFstatus bfRayleighExecute(BFrayleigh     plan,
                           BFarray const* in,
                           BFarray const* out
                           BFsize*        flags) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
	BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
	BF_ASSERT( in->ndim >= 2,                              BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(out->ndim == in->ndim,                       BF_STATUS_INVALID_SHAPE);
	
	BFarray out_flattened, in_flattened;
	if( in->ndim > 2 ) {
		// Keep the first dim but attempt to flatten all others
		unsigned long keep_dims_mask = 0x1;
		keep_dims_mask |= padded_dims_mask(out);
		keep_dims_mask |= padded_dims_mask(in);
		flatten(out, &out_flattened, keep_dims_mask);
		flatten(in,   &in_flattened, keep_dims_mask);
		out = &out_flattened;
		in  =  &in_flattened;
		BF_ASSERT(in_flattened.ndim == out_flattened.ndim,         BF_STATUS_INTERNAL_ERROR);
		BF_ASSERT(in_flattened.ndim == 2,                          BF_STATUS_UNSUPPORTED_SHAPE);
		BF_ASSERT(in_flattened.shape[1] == out_flattened.shape[1], BF_STATUS_INVALID_SHAPE);
	}
	BF_ASSERT( in->shape[1] == plan->nantpol(), BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(out->shape[1] == in->shape[1],    BF_STATUS_INVALID_SHAPE);
	
	BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
	BF_TRY_RETURN(plan->execute(in, out, flags));
}

BFstatus bfRayleighDestroy(BFrayleigh plan) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	delete plan;
	return BF_STATUS_SUCCESS;
}
