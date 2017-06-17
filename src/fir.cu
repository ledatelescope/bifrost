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

template<typename InType, typename OutType>
__global__ void fir_kernel(int                   ncoeff,
                           int                   decim, 
                           int                   ntime, 
                           int                   nstand,
                           double*               coeffs,
                           Complex64*            state0,
                           Complex64*            state1,
                           InType*               d_in,
                           OutType* __restrict__ d_out) {
	int t0 = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x;
	t0 *= decim;
	
	int c, t, s, p;
	Complex64 tempI(0.,0.);
	Complex64 tempO(0.,0.);
	if( t0 < ntime ) {
		for(s=0; s<nstand; s++) {
			for(p=0; p<2; p++) { 
				tempO *= 0.0;
				for(c=0; c<ncoeff; c++) {
					t = t0 - c;
					if( t < 0 ) {
						// Need to seed using the initial state
						tempI = state0[c*nstand*2 + s*2 + p];
					} else {
						// Fully inside the data
						tempI = Complex64(d_in[t*nstand*2*2 + s*2*2 + p*2 + 0], \
						                  d_in[t*nstand*2*2 + s*2*2 + p*2 + 1]);
					}
					tempO += tempI*coeffs[c];
				}
				d_out[t0/decim*nstand*2 + s*2 + p] = tempO;
				
				c = ncoeff - (ntime - t0);
				if( c >= 0 && c < ncoeff ) {
					// Seed the initial state of the next call
					state1[c*nstand*2 + s*2 + p] = Complex64(d_in[t0*nstand*2*2 + s*2*2 + p*2 + 0], \
					                                         d_in[t0*nstand*2*2 + s*2*2 + p*2 + 1]);
				}
			}
		}
	}
}

template<typename InType, typename OutType>
inline void launch_fir_kernel(int          ncoeff,
                              int          decim, 
                              int          ntime, 
                              int          nstand,
                              double*      coeffs,
                              Complex64*   state0,
                              Complex64*   state1,
                              InType*      d_in,
                              OutType*     d_out,
                              cudaStream_t stream=0) {
	ntime /= decim;
	//cout << "LAUNCH for " << nelement << endl;
	dim3 block(512, 1); // TODO: Tune this
	int first = std::min((ntime-1)/block.x+1, 65535u);
	int secnd = std::min((ntime - first*block.x) / first + 1, 65535u);
	if( block.x*first > ntime ) {
		secnd = 1;
	}
	ntime *= decim;
	
	dim3 grid(first, secnd);
	cout << "  Block size is " << block.x << " by " << block.y << endl;
	cout << "  Grid  size is " << grid.x << " by " << grid.y << endl;
	cout << "  Maximum size is " << block.x*grid.x*grid.y << endl;
	if( block.x*grid.x*grid.y >= ntime ) {
		cout << "  -> Valid" << endl;
	}
	
	void* args[] = {&ncoeff,
	                &decim,
	                &ntime, 
	                &nstand,
	                &coeffs,
	                &state0,
	                &state1,
	                &d_in,
	                &d_out};
	cudaLaunchKernel((void*)fir_kernel<InType,OutType>,
	                 grid, block,
	                 &args[0], 0, stream);
}

class BFfir_impl {
	typedef int    IType;
	typedef double FType;
public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
	typedef float  DType;
private:
	IType        _ncoeff;
	IType        _decim;
	IType        _ntime;
	IType        _nstand;
	double*      _coeffs = NULL;
	Complex64*   _state0 = NULL;
	Complex64*   _state1 = NULL;
	IType        _plan_stride;
	Workspace    _plan_storage;
	// TODO: Use something other than Thrust
	thrust::device_vector<char> _dv_plan_storage;
	cudaStream_t _stream;
public:
	BFfir_impl() : _ncoeff(0), _decim(1), _ntime(0), _nstand(0), 
	               _stream(g_cuda_stream) {}
	inline IType ncoeff()   const { return _ncoeff; }
	inline IType decim()    const { return _decim;  }
	inline IType ntime()    const { return _ntime;  }
	inline IType nstand()   const { return _nstand; }
	void init(IType ncoeff, 
	          IType decim,
	          IType ntime,
	          IType nstand) {
		BF_TRACE();
		if( ncoeff == _ncoeff &&
		    decim  == _decim  &&
		    ntime  == _ntime  &&
		    nstand == _nstand) {
			return;
		}
		_ncoeff = ncoeff;
		_decim  = decim;
		_ntime  = ntime;
		_nstand = nstand;
		
		_coeffs = NULL;
		_state0 = NULL;
		_state1 = NULL;
	}
	bool init_plan_storage(void* storage_ptr, BFsize* storage_size) {
		BF_TRACE();
		BF_TRACE_STREAM(_stream);
		enum {
			ALIGNMENT_BYTES = 512,
			ALIGNMENT_ELMTS = ALIGNMENT_BYTES / sizeof(Complex64)
		};
		Workspace workspace(ALIGNMENT_BYTES);
		_plan_stride = round_up(_nstand*2, ALIGNMENT_ELMTS);
		workspace.reserve(_ncoeff*_nstand*2+1, &_state0);
		workspace.reserve(_ncoeff*_nstand*2+1, &_state1);
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
		Complex64* cstate;
		cstate = (Complex64*) malloc(sizeof(Complex64)*_ncoeff*_nstand*2);
		memset(cstate,  0, sizeof(Complex64)*_ncoeff*_nstand*2);
		BF_CHECK_CUDA_EXCEPTION( cudaMemcpyAsync(_state0,
		                                         cstate,
		                                         sizeof(Complex64)*_ncoeff*_nstand*2,
		                                         cudaMemcpyHostToDevice,
		                                         _stream),
		                         BF_STATUS_MEM_OP_FAILED );
		BF_CHECK_CUDA_EXCEPTION( cudaMemcpyAsync(_state1,
		                                         _state0,
		                                         sizeof(Complex64)*_ncoeff*_nstand*2,
		                                         cudaMemcpyDeviceToDevice,
		                                         _stream),
		                         BF_STATUS_MEM_OP_FAILED );
		BF_CHECK_CUDA_EXCEPTION( cudaStreamSynchronize(_stream),
		                         BF_STATUS_DEVICE_ERROR );
		free(cstate);
		
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
	}
	void execute(BFarray const* in,
	             BFarray const* out) {
		BF_TRACE();
		BF_TRACE_STREAM(_stream);
		BF_ASSERT_EXCEPTION(_coeffs != NULL, BF_STATUS_INVALID_STATE);
		BF_ASSERT_EXCEPTION(_state0 != NULL,  BF_STATUS_INVALID_STATE);
		BF_ASSERT_EXCEPTION(_state1 != NULL,  BF_STATUS_INVALID_STATE);
		BF_ASSERT_EXCEPTION(out->dtype == BF_DTYPE_CF32 || \
		                    out->dtype == BF_DTYPE_CF64,     BF_STATUS_UNSUPPORTED_DTYPE);
		
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
		
#define LAUNCH_BEAMFORMER_KERNEL(IterType,OterType) \
		launch_fir_kernel(_ncoeff, _decim, _ntime, _nstand, \
		                  _coeffs, _state0, _state1, \
		                  (IterType)in->data, (OterType)out->data, \
		                  _stream)
		
		switch( in->dtype ) {
			case BF_DTYPE_CI8:
				switch( out->dtype ) {
					case BF_DTYPE_CF32: LAUNCH_BEAMFORMER_KERNEL(int8_t*, Complex32*);  break;
					case BF_DTYPE_CF64: LAUNCH_BEAMFORMER_KERNEL(int8_t*, Complex64*);  break;
					default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
				};
				break;
			case BF_DTYPE_CI16:
				switch( out->dtype ) {
					case BF_DTYPE_CF32: LAUNCH_BEAMFORMER_KERNEL(int16_t*, Complex32*); break;
					case BF_DTYPE_CF64: LAUNCH_BEAMFORMER_KERNEL(int16_t*, Complex64*); break;
					default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
				}
				break;
			case BF_DTYPE_CI32:
				switch( out->dtype ) {
					case BF_DTYPE_CF32: LAUNCH_BEAMFORMER_KERNEL(int32_t*, Complex32*); break;
					case BF_DTYPE_CF64: LAUNCH_BEAMFORMER_KERNEL(int32_t*, Complex64*); break;
					default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
				}
				break;
			case BF_DTYPE_CI64:
				switch( out->dtype ) {
					case BF_DTYPE_CF32: LAUNCH_BEAMFORMER_KERNEL(int64_t*, Complex32*); break;
					case BF_DTYPE_CF64: LAUNCH_BEAMFORMER_KERNEL(int64_t*, Complex64*); break;
					default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
				}
				break;
			case BF_DTYPE_CF32:
				switch( out->dtype ) {
					case BF_DTYPE_CF32: LAUNCH_BEAMFORMER_KERNEL(float*, Complex32*);   break;
					case BF_DTYPE_CF64: LAUNCH_BEAMFORMER_KERNEL(float*, Complex64*);   break;
					default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
				}
				break;
			case BF_DTYPE_CF64:
				switch( out->dtype ) {
					case BF_DTYPE_CF32: LAUNCH_BEAMFORMER_KERNEL(double*, Complex32*);  break;
					case BF_DTYPE_CF64: LAUNCH_BEAMFORMER_KERNEL(double*, Complex64*);  break;
					default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
				}
				break;
			default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
		}
#undef LAUNCH_BEAMFORMER_KERNEL
		
		BF_CHECK_CUDA_EXCEPTION( cudaMemcpyAsync(_state0,
		                                         _state1,
		                                         sizeof(Complex64)*_ncoeff*_nstand*2,
		                                         cudaMemcpyDeviceToDevice,
		                                         _stream),
		                         BF_STATUS_MEM_OP_FAILED );
		
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
	}
	void set_stream(cudaStream_t stream) {
		_stream = stream;
	}
};

BFstatus bfFIRCreate(BFfir* plan_ptr) {
	BF_TRACE();
	BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*plan_ptr = new BFfir_impl(),
	                   *plan_ptr = 0);
}
// **TODO: Passing 'BFarray const* in' here could replace nchan, f0, df and space if BFarray included dimension scales
//           Also, could potentially set the output dimension scales (dm0, ddm)
//           OR, could just leave these to higher-level wrappers (e.g., Python)
//             This might be for the best in the short term
BFstatus bfFIRInit(BFfir   plan,
                   BFsize  ncoeff,
                   BFsize  decim,
                   BFsize  nstand,
                   BFsize  ntime,
                   BFspace space,
                   void*   plan_storage,
                   BFsize* plan_storage_size) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(decim == 1, BF_STATUS_UNSUPPORTED);
	BF_ASSERT(space_accessible_from(space, BF_SPACE_CUDA),
	          BF_STATUS_UNSUPPORTED_SPACE);
	BF_TRY(plan->init(ncoeff, decim, nstand, ntime));
	BF_TRY_RETURN(plan->init_plan_storage(plan_storage, plan_storage_size));
}
BFstatus bfFIRSetStream(BFfir        plan,
                        void const*  stream) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}
BFstatus bfFIRSetCoeffs(BFfir          plan, 
                        BFarray const* coeffs) {
	BF_ASSERT(coeffs,                                          BF_STATUS_INVALID_POINTER);
	BF_ASSERT(coeffs->shape[coeffs->ndim-1] == plan->ncoeff(), BF_STATUS_INVALID_SHAPE  );
	
	// TODO: BF_ASSERT(...);
	BF_ASSERT(space_accessible_from(coeffs->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
	BF_TRY_RETURN(plan->set_coeffs(coeffs));
}
BFstatus bfFIRResetState(BFfir plan) {
	BF_TRY_RETURN(plan->reset_state());
}
BFstatus bfFIRExecute(BFfir          plan,
                      BFarray const* in,
                      BFarray const* out) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
	BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
	BF_ASSERT( in->shape[ in->ndim-3] == plan->ntime(),  BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( in->shape[ in->ndim-2] == plan->nstand(), BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( in->shape[ in->ndim-1] == 2,              BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( out->shape[out->ndim-3] == in->shape[in->ndim-3]/plan->decim(), BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( out->shape[out->ndim-2] == in->shape[in->ndim-2],               BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( out->shape[out->ndim-1] == in->shape[in->ndim-1],               BF_STATUS_INVALID_SHAPE);
	
	// TODO: BF_ASSERT(...);
	BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
	BF_TRY_RETURN(plan->execute(in, out));
}

BFstatus bfFIRDestroy(BFfir plan) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	delete plan;
	return BF_STATUS_SUCCESS;
}
