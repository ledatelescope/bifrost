/*
 * Copyright (c) 2017, The Bifrost Authors. All rights reserved.
 * Copyright (c) 2017, The University of New Mexico. All rights reserved.
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

#include <vector>
#include <map>
#include <string>

// HACK TESTING
#include <iostream>
using std::cout;
using std::endl;

__device__
inline double fetch_f64(cudaTextureObject_t tex,
                        int                 idx) {
    uint2 p = tex1Dfetch<uint2>(tex, idx);
    return __hiloint2double(p.y, p.x);
}

template<typename InType, typename OutType, bool IsComplex>
__global__ void fir_kernel(unsigned int               ncoeff,
                           unsigned int               decim, 
                           unsigned int               ntime, 
                           unsigned int               nantpol,
                           uint8_t                    dfactor,
                           cudaTextureObject_t coeffs,
                           double*                    state0,
                           double*                    state1,
                           const InType* __restrict__ d_in,
                           OutType* __restrict__      d_out) {
	int a = threadIdx.x + blockIdx.x*blockDim.x;
	
	int c, t, t0;
	double tempI, tempO;
	if( a < nantpol ) {
		t0 = threadIdx.y + (blockIdx.y + blockIdx.z*gridDim.y)*blockDim.y;
		t0 *= decim;
		if( t0 < ntime ) {
			tempO = 0.0;
			for(c=0; c<ncoeff; c++) {
				t = t0 - ncoeff + c + 1;
				if( t < 0 ) {
					// Need to seed using the initial state
					tempI = state0[(ncoeff+t)*nantpol + a];
				} else {
					// Fully inside the data
          tempI = (double) d_in[t*nantpol + a];
				}
				tempO += tempI*fetch_f64(coeffs, nantpol/dfactor*c + a/dfactor);
			}
      d_out[t0/decim*nantpol + a] = OutType(tempO);
			
			for(t=t0; t<t0+decim; t++) {
				c = ncoeff - (ntime - t);
				if( c >= 0 && c < ncoeff && t < ntime) {
					// Seed the initial state of the next call
          state1[c*nantpol + a] = (double) d_in[t*nantpol + a];
				}
			}
		}
	}
}

template<typename InType, typename OutType, bool IsComplex>
inline void launch_fir_kernel(unsigned int ncoeff,
                              unsigned int decim, 
                              unsigned int ntime, 
                              unsigned int nantpol,
                              double*      coeffs,
                              double*      state0,
                              double*      state1,
                              InType*      d_in,
                              OutType*     d_out,
                              cudaStream_t stream=0) {
  uint8_t dfactor = 1+IsComplex;
  nantpol *= dfactor;
  
	//cout << "LAUNCH for " << nelement << endl;
	dim3 block(std::min(256u, nantpol), 256u/std::min(256u, nantpol));
	int first = std::min((nantpol-1)/block.x+1, 65535u);
	int secnd = std::min((ntime/decim-1)/block.y+1, 65535u);
	int third = std::min((ntime/decim-secnd*block.y-1)/secnd+2, 65535u);
	if( block.y*secnd >= ntime/decim ) {
		third = 1;
	}
	
	dim3 grid(first, secnd, third);
	/*
	cout << "  Block size is " << block.x << " by " << block.y << endl;
	cout << "  Grid  size is " << grid.x << " by " << grid.y << " by " << grid.z << endl;
	cout << "  Maximum size is " << block.y*grid.y*grid.z << endl;
	if( block.y*grid.y*grid.z >= ntime ) {
		cout << "  -> Valid" << endl;
	}
	*/
	
  // Create texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = coeffs;
  resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
  resDesc.res.linear.desc.x = 32;
  resDesc.res.linear.desc.y = 32;
  resDesc.res.linear.desc.z = 0;
  resDesc.res.linear.desc.w = 0;
  resDesc.res.linear.sizeInBytes = nantpol/dfactor*ncoeff*sizeof(double);
  
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  
  cudaTextureObject_t coeffs_tex;
  BF_CHECK_CUDA_EXCEPTION(cudaCreateTextureObject(&coeffs_tex, &resDesc, &texDesc, NULL),
                          BF_STATUS_INTERNAL_ERROR);

	void* args[] = {&ncoeff,
	                &decim,
	                &ntime, 
	                &nantpol,
                  &dfactor,
	                &coeffs_tex,
	                &state0,
	                &state1,
	                &d_in,
	                &d_out};
	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)fir_kernel<InType,OutType,IsComplex>,
	                                         grid, block,
	                                         &args[0], 0, stream),
	                        BF_STATUS_INTERNAL_ERROR);
  
  BF_CHECK_CUDA_EXCEPTION(cudaDestroyTextureObject(coeffs_tex),
                            BF_STATUS_INTERNAL_ERROR);
}

class BFfir_impl {
	typedef int          IType;
	typedef unsigned int UType;
	typedef double       FType;
public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
	typedef float  DType;
private:
	UType        _ncoeff;
	UType        _decim;
	UType        _nantpol;
	double*      _coeffs = NULL;
	double*      _state0 = NULL;
	double*      _state1 = NULL;
	IType        _plan_stride;
	Workspace    _plan_storage;
	// TODO: Use something other than Thrust
	thrust::device_vector<char> _dv_plan_storage;
	cudaStream_t _stream;
public:
	BFfir_impl() : _coeffs(NULL), _decim(1), _stream(g_cuda_stream) {}
	inline UType ncoeff()   const { return _ncoeff;  }
	inline UType decim()    const { return _decim;   }
	inline UType nantpol()  const { return _nantpol; }
	void init(UType ncoeffs,
	          UType nantpol, 
	          UType decim) {
		BF_TRACE();
		_decim   = decim;
		_ncoeff  = ncoeffs;
		_nantpol = nantpol;
		
		_state0 = NULL;
		_state1 = NULL;
	}
	bool init_plan_storage(void* storage_ptr, BFsize* storage_size) {
		BF_TRACE();
		BF_TRACE_STREAM(_stream);
		enum {
			ALIGNMENT_BYTES = 512,
			ALIGNMENT_ELMTS = ALIGNMENT_BYTES / sizeof(double)
		};
		Workspace workspace(ALIGNMENT_BYTES);
		_plan_stride = round_up(_nantpol, ALIGNMENT_ELMTS);
		workspace.reserve(2*_ncoeff*_nantpol+1, &_state0);
		workspace.reserve(2*_ncoeff*_nantpol+1, &_state1);
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
	void set_coeffs(BFarray const* coeffs) { 
		BF_TRACE();
		BF_TRACE_STREAM(_stream);
		BF_ASSERT_EXCEPTION(coeffs->dtype == BF_DTYPE_F64, BF_STATUS_UNSUPPORTED_DTYPE);
		
		_coeffs = (double*) coeffs->data;
		this->reset_state();
	}
	void reset_state() {
		BF_ASSERT_EXCEPTION(_state0 != NULL,  BF_STATUS_INVALID_STATE);
		BF_ASSERT_EXCEPTION(_state1 != NULL,  BF_STATUS_INVALID_STATE);
		
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
		
		// Reset the state
		BF_CHECK_CUDA_EXCEPTION( cudaMemsetAsync(_state0,
		                                         0,
		                                         sizeof(Complex64)*_ncoeff*_nantpol,
		                                         _stream),
		                         BF_STATUS_MEM_OP_FAILED );
		BF_CHECK_CUDA_EXCEPTION( cudaMemsetAsync(_state1,
		                                         0,
		                                         sizeof(Complex64)*_ncoeff*_nantpol,
		                                         _stream),
		                         BF_STATUS_MEM_OP_FAILED );
		BF_CHECK_CUDA_EXCEPTION( cudaStreamSynchronize(_stream),
		                         BF_STATUS_DEVICE_ERROR );
		
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
	}
	void execute(BFarray const* in,
	             BFarray const* out) {
		BF_TRACE();
		BF_TRACE_STREAM(_stream);
		BF_ASSERT_EXCEPTION(_coeffs != NULL, BF_STATUS_INVALID_STATE);
		BF_ASSERT_EXCEPTION(_state0 != NULL, BF_STATUS_INVALID_STATE);
		BF_ASSERT_EXCEPTION(_state1 != NULL, BF_STATUS_INVALID_STATE);
		BF_ASSERT_EXCEPTION(out->dtype == BF_DTYPE_F32 || \
                        out->dtype == BF_DTYPE_F64 || \
                        out->dtype == BF_DTYPE_CF32 || \
		                    out->dtype == BF_DTYPE_CF64,     BF_STATUS_UNSUPPORTED_DTYPE);
		
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
		
#define LAUNCH_FIR_KERNEL(IterType,OterType,IsComplex) \
		launch_fir_kernel<IterType,OterType,IsComplex>(_ncoeff, _decim, in->shape[0], _nantpol, \
                              		                 _coeffs, _state0, _state1, \
                              		                 (IterType*)in->data, (OterType*)out->data, \
                              		                 _stream)
		
		switch( in->dtype ) {
      case BF_DTYPE_I8:
        switch( out->dtype ) {
          case BF_DTYPE_F32: LAUNCH_FIR_KERNEL(int8_t, float, false);  break;
          case BF_DTYPE_F64: LAUNCH_FIR_KERNEL(int8_t, double, false);  break;
          default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        };
        break;
			case BF_DTYPE_CI8:
				switch( out->dtype ) {
					case BF_DTYPE_CF32: LAUNCH_FIR_KERNEL(int8_t, float, true);  break;
					case BF_DTYPE_CF64: LAUNCH_FIR_KERNEL(int8_t, double, true);  break;
					default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
				};
				break;
      case BF_DTYPE_I16:
        switch( out->dtype ) {
          case BF_DTYPE_F32: LAUNCH_FIR_KERNEL(int16_t, float, false);  break;
          case BF_DTYPE_F64: LAUNCH_FIR_KERNEL(int16_t, double, false);  break;
          default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        };
        break;
			case BF_DTYPE_CI16:
				switch( out->dtype ) {
					case BF_DTYPE_CF32: LAUNCH_FIR_KERNEL(int16_t, float, true); break;
					case BF_DTYPE_CF64: LAUNCH_FIR_KERNEL(int16_t, double, true); break;
					default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
				}
				break;
      case BF_DTYPE_I32:
        switch( out->dtype ) {
          case BF_DTYPE_F32: LAUNCH_FIR_KERNEL(int32_t, float, false);  break;
          case BF_DTYPE_F64: LAUNCH_FIR_KERNEL(int32_t, double, false);  break;
          default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        };
        break;
			case BF_DTYPE_CI32:
				switch( out->dtype ) {
					case BF_DTYPE_CF32: LAUNCH_FIR_KERNEL(int32_t, float, true); break;
					case BF_DTYPE_CF64: LAUNCH_FIR_KERNEL(int32_t, double, true); break;
					default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
				}
				break;
      case BF_DTYPE_I64:
        switch( out->dtype ) {
          case BF_DTYPE_F32: LAUNCH_FIR_KERNEL(int64_t, float, false);  break;
          case BF_DTYPE_F64: LAUNCH_FIR_KERNEL(int64_t, double, false);  break;
          default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        };
        break;
			case BF_DTYPE_CI64:
				switch( out->dtype ) {
					case BF_DTYPE_CF32: LAUNCH_FIR_KERNEL(int64_t, float, true); break;
					case BF_DTYPE_CF64: LAUNCH_FIR_KERNEL(int64_t, double, true); break;
					default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
				}
				break;
      case BF_DTYPE_F32:
        switch( out->dtype ) {
          case BF_DTYPE_F32: LAUNCH_FIR_KERNEL(float, float, false);  break;
          case BF_DTYPE_F64: LAUNCH_FIR_KERNEL(float, double, false);  break;
          default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        };
        break;
			case BF_DTYPE_CF32:
				switch( out->dtype ) {
					case BF_DTYPE_CF32: LAUNCH_FIR_KERNEL(float, float, true);   break;
					case BF_DTYPE_CF64: LAUNCH_FIR_KERNEL(float, double, true);   break;
					default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
				}
				break;
      case BF_DTYPE_F64:
        switch( out->dtype ) {
          case BF_DTYPE_F32: LAUNCH_FIR_KERNEL(double, float, false);  break;
          case BF_DTYPE_F64: LAUNCH_FIR_KERNEL(double, double, false);  break;
          default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        };
        break;
			case BF_DTYPE_CF64:
				switch( out->dtype ) {
					case BF_DTYPE_CF32: LAUNCH_FIR_KERNEL(double, float, true);  break;
					case BF_DTYPE_CF64: LAUNCH_FIR_KERNEL(double, double, true);  break;
					default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
				}
				break;
			default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
		}
#undef LAUNCH_FIR_KERNEL
		
		BF_CHECK_CUDA_EXCEPTION( cudaMemcpyAsync(_state0,
		                                         _state1,
		                                         sizeof(Complex64)*_ncoeff*_nantpol,
		                                         cudaMemcpyDeviceToDevice,
		                                         _stream),
		                         BF_STATUS_MEM_OP_FAILED );
		
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
	}
	void set_stream(cudaStream_t stream) {
		_stream = stream;
	}
};

BFstatus bfFirCreate(BFfir* plan_ptr) {
	BF_TRACE();
	BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*plan_ptr = new BFfir_impl(),
	                   *plan_ptr = 0);
}

BFstatus bfFirInit(BFfir          plan,
                   BFarray const* coeffs, 
                   BFsize         decim,
                   BFspace        space,
                   void*          plan_storage,
                   BFsize*        plan_storage_size) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(coeffs,            BF_STATUS_INVALID_POINTER);
	BF_ASSERT(coeffs->ndim >= 2, BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(space_accessible_from(coeffs->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
	BF_ASSERT(space_accessible_from(space, BF_SPACE_CUDA),
	          BF_STATUS_UNSUPPORTED_SPACE);
	
	// Discover the dimensions of the FIR coefficients.  This uses the 
	// first dimension to set the number of coefficients.  All following
	// dimensions are merged together to get the the number of ant/pols.
	unsigned int ncoeff, nantpols;
	ncoeff = coeffs->shape[0];
	nantpols = 1;
	for(int i=1; i<coeffs->ndim; ++i) {
		nantpols *= coeffs->shape[i];
	}
	BF_TRY(plan->init(ncoeff, nantpols, decim));
	BF_TRY(plan->init_plan_storage(plan_storage, plan_storage_size));
	BF_TRY_RETURN(plan->set_coeffs(coeffs));
}
BFstatus bfFirSetStream(BFfir        plan,
                        void const*  stream) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}
BFstatus bfFirSetCoeffs(BFfir          plan, 
                        BFarray const* coeffs) {
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(coeffs,            BF_STATUS_INVALID_POINTER);
	BF_ASSERT(coeffs->ndim >= 2, BF_STATUS_INVALID_SHAPE  );
	
	BFarray coeffs_flattened;
	if( coeffs->ndim > 2 ) {
		// Keep the first dim but attempt to flatten all others
		unsigned long keep_dims_mask = 0x1;
		keep_dims_mask |= padded_dims_mask(coeffs);
		flatten(coeffs,   &coeffs_flattened, keep_dims_mask);
		coeffs = &coeffs_flattened;
		BF_ASSERT(coeffs_flattened.ndim == 2, BF_STATUS_UNSUPPORTED_SHAPE);
	}
	BF_ASSERT(coeffs->shape[coeffs->ndim-2] == plan->ncoeff(),  BF_STATUS_INVALID_SHAPE  );
	BF_ASSERT(coeffs->shape[coeffs->ndim-1] == plan->nantpol(), BF_STATUS_INVALID_SHAPE  );
	
	BF_ASSERT(space_accessible_from(coeffs->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
	BF_TRY_RETURN(plan->set_coeffs(coeffs));
}
BFstatus bfFirResetState(BFfir plan) {
	BF_TRY_RETURN(plan->reset_state());
}
BFstatus bfFirExecute(BFfir          plan,
                      BFarray const* in,
                      BFarray const* out) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
	BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
	BF_ASSERT( in->ndim >= 2,                              BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(out->ndim == in->ndim,                       BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( in->shape[0] %  plan->decim() == 0,         BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(out->shape[0] == in->shape[0]/plan->decim(), BF_STATUS_INVALID_SHAPE);
	
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
	BF_TRY_RETURN(plan->execute(in, out));
}

BFstatus bfFirDestroy(BFfir plan) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	delete plan;
	return BF_STATUS_SUCCESS;
}
