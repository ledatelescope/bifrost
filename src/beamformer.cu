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

#include <bifrost/beamformer.h>
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

template<typename DType>
__global__
void delay_to_phase_kernel(int        nchan, 
                           int        nstand, 
                           double     freq0, 
                           double     freqStep, 
                           DType*     delays, 
                           Complex64* prots) {
	int c0 = threadIdx.x + blockIdx.x*blockDim.x;
	int s0 = threadIdx.y + blockIdx.y*blockDim.y;
	
	Complex64 CI(0.,1.);
	
	if( c0 < nchan && s0 < nstand ) {
#pragma unroll
		for(int p=0; p<2; p++) {
			prots[c0*nstand*2 + s0*2 + p] = exp(-2*PI_DOUBLE*CI*(freq0+freqStep*c0)*((double) delays[s0*2 + p]));
		}
	}
}

template<typename DType>
inline
void launch_delay_to_phase_kernel(int          nchan,
                                  int          nstand, 
                                  double       freq0, 
                                  double       freqStep, 
                                  DType*       delays,
                                  Complex64*   prots,
                                  cudaStream_t stream=0) {
	//cout << "LAUNCH for " << nchan << " by " << nstand << endl;
	dim3 block(32, 16); // TODO: Tune this
	dim3 grid(std::min((nchan-1)/block.x+1, 65535u),
	          std::min((nstand-1)/block.y+1, 65535u));
	/*
	cout << "  Block size is " << block.x << " by " << block.y << endl;
	cout << "  Grid  size is " << grid.x << " by " << grid.y << endl;
	*/
	
	void* args[] = {&nchan,
	                &nstand, 
	                &freq0, 
	                &freqStep, 
	                &delays, 
	                &prots};
	cudaLaunchKernel((void*)delay_to_phase_kernel<DType>,
	                 grid, block,
	                 &args[0], 0, stream);
}

template<typename DType>
__global__
void beamformer_kernel(int                     ntime, 
                       int                     nchan, 
                       int                     nstand,
                       Complex64*              prots,
                       double*                 gains,
                       DType*                  d_in,
                       Complex32* __restrict__ d_out) {
	int t0 = threadIdx.x + blockIdx.x*blockDim.x;
	int c0 = threadIdx.y + blockIdx.y*blockDim.y;
	
	Complex64 CI(0.,1.);
	Complex64 tempP(0.,0.);
	Complex64 beamX(0.,0.);
	Complex64 beamY(0.,0.);
	
	if( t0 < ntime && c0 < nchan) {
		beamX *= 0.0;
		beamY *= 0.0;
		
		for(int s=0; s<nstand; s++) {
#pragma unroll
			for(int p=0; p<2; p++) { 
				tempP  = Complex64(d_in[t0*nchan*nstand*2*2 + c0*nstand*2*2 + s*2*2 + p*2 + 0], \
							    d_in[t0*nchan*nstand*2*2 + c0*nstand*2*2 + s*2*2 + p*2 + 1]);
				tempP *= prots[ c0*nstand*2 + s*2 + p];
				beamX += tempP * gains[s*2*2 + p*2 + 0];
				beamY += tempP * gains[s*2*2 + p*2 + 1];
			}
		}
		
		d_out[t0*nchan*2 + c0*2 + 0] = Complex32(beamX.real, beamX.imag);
		d_out[t0*nchan*2 + c0*2 + 1] = Complex32(beamY.real, beamY.imag);
	}
}

template<typename DType>
inline
void launch_beamformer_kernel(int          ntime,
                              int          nchan,
                              int          nstand, 
                              Complex64*   prots,
                              double*      gains,
                              DType*       d_in,
                              Complex32*   d_out,
                              cudaStream_t stream=0) {
	// cout << "LAUNCH for " << ntime << " by " << nchan << " by " << nstand << endl;
	dim3 block(32, 16); // TODO: Tune this
	dim3 grid(std::min((ntime-1)/block.x+1, 65535u),
	          std::min((nchan-1)/block.y+1, 65535u));
	/*
	cout << "  Block size is " << block.x << " by " << block.y << endl;
	cout << "  Grid  size is " << grid.x << " by " << grid.y << endl;
	*/
	
	void* args[] = {&ntime,
	                &nchan,
	                &nstand, 
	                &prots,
	                &gains,
	                &d_in,
	                &d_out};
	cudaLaunchKernel((void*)beamformer_kernel<DType>,
	                 grid, block,
	                 &args[0], 0, stream);
}

#define BIFROST_TO_CFLOAT(X)  ((float)  ((X>>4)&0xF-2*((X>>4)&8)) + CI*((float)  ((X&0xF)-2*(X&8))))
#define BIFROST_TO_CDOUBLE(X) ((double) ((X>>4)&0xF-2*((X>>4)&8)) + CI*((double) ((X&0xF)-2*(X&8))))

__global__
void beamformer_kernel_CI4(int                     ntime, 
                           int                     nchan, 
                           int                     nstand,
                           Complex64*              prots,
                           double*                 gains,
                           uint8_t*                d_in,
                           Complex32* __restrict__ d_out) {
	int t0 = threadIdx.x + blockIdx.x*blockDim.x;
	int c0 = threadIdx.y + blockIdx.y*blockDim.y;
	
	Complex64 CI(0.,1.);
	Complex64 tempP(0.,0.);
	Complex64 beamX(0.,0.);
	Complex64 beamY(0.,0.);
	
	if( t0 < ntime && c0 < nchan) {
		beamX = 0.0 + CI*0.0;
		beamY = 0.0 + CI*0.0;
		
		for(int s=0; s<nstand; s++) {
#pragma unroll
			for(int p=0; p<2; p++) { 
				tempP  = (Complex64) BIFROST_TO_CDOUBLE(d_in[t0*nchan*nstand*2 + c0*nstand*2 + s*2 + p]);
				tempP *= prots[ c0*nstand*2 + s*2 + p];
				beamX += tempP * gains[s*2*2 + p*2 + 0];
				beamY += tempP * gains[s*2*2 + p*2 + 1];
			}
		}
		
		d_out[t0*nchan*2 + c0*2 + 0] = Complex32(beamX.real, beamX.imag);
		d_out[t0*nchan*2 + c0*2 + 1] = Complex32(beamY.real, beamY.imag);
	}
}

inline
void launch_beamformer_kernel_CI4(int          ntime,
                                  int          nchan,
                                  int          nstand, 
                                  Complex64*   prots,
                                  double*      gains,
                                  uint8_t*     d_in,
                                  Complex32*   d_out,
                                  cudaStream_t stream=0) {
	// cout << "LAUNCH for " << ntime << " by " << nchan << " by " << nstand << endl;
	dim3 block(32, 16); // TODO: Tune this
	dim3 grid(std::min((ntime-1)/block.x+1, 65535u),
	          std::min((nchan-1)/block.y+1, 65535u));
	/*
	cout << "  Block size is " << block.x << " by " << block.y << endl;
	cout << "  Grid  size is " << grid.x << " by " << grid.y << endl;
	*/
	
	void* args[] = {&ntime,
	                &nchan,
	                &nstand, 
	                &prots,
	                &gains,
	                &d_in,
	                &d_out};
	cudaLaunchKernel((void*)beamformer_kernel_CI4,
	                 grid, block,
	                 &args[0], 0, stream);
}

class BFbeamformer_impl {
	typedef int    IType;
	typedef double FType;
public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
	typedef float  DType;
private:
	IType        _ntime;
	IType        _nchan;
	IType        _nstand;
	Complex64*   _prots;
	double*      _gains;
	cudaStream_t _stream;
public:
	BFbeamformer_impl() : _ntime(0), _nchan(0), _nstand(0),
	                _stream(g_cuda_stream) {}
	inline IType ntime()     const { return _ntime; }
	inline IType nchan()     const { return _nchan; }
	inline IType nstand()    const {return _nstand; }
	void init(IType ntime,
	          IType nchan,
	          IType nstand) {
		BF_TRACE();
		if( ntime  == _ntime &&
		    nchan  == _nchan &&
		    nstand == _nstand ) {
			return;
		}
		_ntime  = ntime;
		_nchan  = nchan;
		_nstand = nstand;
	}
	void set_delays(double freq0, double freqStep, BFarray const* delays, BFarray const* prots) {
		BF_TRACE();
		BF_TRACE_STREAM(_stream);
		BF_ASSERT_EXCEPTION(prots->dtype == BF_DTYPE_CF64, BF_STATUS_UNSUPPORTED_DTYPE);
		
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
#define LAUNCH_DELAY_KERNEL(IterType) \
	launch_delay_to_phase_kernel(_nchan, _nstand, \
		                        freq0, freqStep, \
		                        (IterType)delays->data, (Complex64*)prots->data, \
		                        _stream)
		switch( delays->dtype ) {
			case BF_DTYPE_F32: LAUNCH_DELAY_KERNEL(float*);  break;
			case BF_DTYPE_F64: LAUNCH_DELAY_KERNEL(double*); break;
			default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
		}
#undef LAUNCH_DELAY_KERNEL
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
		_prots = (Complex64*) prots->data;
	}
	void set_phase_rotators(BFarray const* prots) { 
		BF_TRACE();
		BF_TRACE_STREAM(_stream);
		BF_ASSERT_EXCEPTION(prots->dtype == BF_DTYPE_CF64, BF_STATUS_UNSUPPORTED_DTYPE);
		_prots = (Complex64*) prots->data;
	}
	void set_gains(BFarray const* gains) {
		BF_TRACE();
		BF_TRACE_STREAM(_stream);
		BF_ASSERT_EXCEPTION(gains->dtype == BF_DTYPE_F64, BF_STATUS_UNSUPPORTED_DTYPE);
		_gains = (double*) gains->data;
	}
	void execute(BFarray const* in,
	             BFarray const* out) {
		BF_TRACE();
		BF_TRACE_STREAM(_stream);
		BF_ASSERT_EXCEPTION(_prots != NULL, BF_STATUS_INVALID_STATE);
		BF_ASSERT_EXCEPTION(_gains != NULL, BF_STATUS_INVALID_STATE);
		BF_ASSERT_EXCEPTION(out->dtype == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
		
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
#define LAUNCH_BEAMFORMER_KERNEL(IterType) \
		launch_beamformer_kernel(_ntime, _nchan, _nstand, \
		                        _prots, _gains, \
		                        (IterType)in->data, (Complex32*)out->data, \
		                        _stream)
		
		switch( in->dtype ) {
			case BF_DTYPE_CI4:  launch_beamformer_kernel_CI4(_ntime, _nchan, _nstand, \
		                                                      _prots, _gains, \
		                                                      (uint8_t*)in->data, (Complex32*)out->data, \
		                                                      _stream);   break;
			case BF_DTYPE_CI8:  LAUNCH_BEAMFORMER_KERNEL(int8_t*);  break;
			case BF_DTYPE_CI16: LAUNCH_BEAMFORMER_KERNEL(int16_t*); break;
			case BF_DTYPE_CI32: LAUNCH_BEAMFORMER_KERNEL(int32_t*); break;
			case BF_DTYPE_CI64: LAUNCH_BEAMFORMER_KERNEL(int64_t*); break;
			case BF_DTYPE_CF32: LAUNCH_BEAMFORMER_KERNEL(float*);   break;
			case BF_DTYPE_CF64: LAUNCH_BEAMFORMER_KERNEL(double*);  break;
			default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
		}
#undef LAUNCH_BEAMFORMER_KERNEL
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
	}
	void set_stream(cudaStream_t stream) {
		_stream = stream;
	}
};

BFstatus bfBeamformerCreate(BFbeamformer* plan_ptr) {
	BF_TRACE();
	BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*plan_ptr = new BFbeamformer_impl(),
	                   *plan_ptr = 0);
}
// **TODO: Passing 'BFarray const* in' here could replace nchan, f0, df and space if BFarray included dimension scales
//           Also, could potentially set the output dimension scales (dm0, ddm)
//           OR, could just leave these to higher-level wrappers (e.g., Python)
//             This might be for the best in the short term
BFstatus bfBeamformerInit(BFbeamformer plan,
                          BFsize       ntime,
                          BFsize       nchan,
                          BFsize       nstand,
                          BFspace      space) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(space_accessible_from(space, BF_SPACE_CUDA),
	          BF_STATUS_UNSUPPORTED_SPACE);
	BF_TRY_RETURN(plan->init(ntime, nchan, nstand));
}
BFstatus bfBeamformerSetStream(BFbeamformer plan,
                               void const*  stream) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}
BFstatus bfBeamformerSetDelays(BFbeamformer   plan, 
                               double         freq0, 
                               double         freqStep, 
                               BFarray const* delays,
                               BFarray const* prots) {
	BF_ASSERT(delays,  BF_STATUS_INVALID_POINTER);
	BF_ASSERT(delays->shape[delays->ndim-2] == plan->nstand(), BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(delays->shape[delays->ndim-1] == 2             , BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(prots,   BF_STATUS_INVALID_POINTER);
	BF_ASSERT(prots->shape[prots->ndim-3] == plan->nchan() , BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(prots->shape[prots->ndim-2] == plan->nstand(), BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(prots->shape[prots->ndim-1] == 2             , BF_STATUS_INVALID_SHAPE);
	
	// TODO: BF_ASSERT(...);
	BF_ASSERT(space_accessible_from(delays->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
	BF_ASSERT(space_accessible_from(prots->space , BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
	BF_TRY_RETURN(plan->set_delays(freq0, freqStep, delays, prots));
}
BFstatus bfBeamformerSetPhaseRotators(BFbeamformer   plan, 
                                      BFarray const* prots) {
	BF_ASSERT(prots,   BF_STATUS_INVALID_POINTER);
	BF_ASSERT(prots->shape[prots->ndim-3] == plan->nchan(),  BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(prots->shape[prots->ndim-2] == plan->nstand(), BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(prots->shape[prots->ndim-1] == 2             , BF_STATUS_INVALID_SHAPE);
	
	// TODO: BF_ASSERT(...);
	BF_ASSERT(space_accessible_from(prots->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
	BF_TRY_RETURN(plan->set_phase_rotators(prots));
}
BFstatus bfBeamformerSetGains(BFbeamformer   plan, 
                              BFarray const* gains) {
	BF_ASSERT(gains,   BF_STATUS_INVALID_POINTER);
	BF_ASSERT(gains->shape[gains->ndim-3] == plan->nstand(), BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(gains->shape[gains->ndim-2] == 2             , BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(gains->shape[gains->ndim-1] == 2             , BF_STATUS_INVALID_SHAPE);
	
	// TODO: BF_ASSERT(...);
	BF_ASSERT(space_accessible_from(gains->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
	BF_TRY_RETURN(plan->set_gains(gains));
}
BFstatus bfBeamformerExecute(BFbeamformer   plan,
                             BFarray const* in,
                             BFarray const* out) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
	BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
	BF_ASSERT( in->shape[ in->ndim-4] == plan->ntime(),     BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( in->shape[ in->ndim-3] == plan->nchan(),     BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( in->shape[ in->ndim-2] == plan->nstand(),    BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( in->shape[ in->ndim-1] == 2,                 BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( out->shape[out->ndim-4] == in->shape[in->ndim-4], BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( out->shape[out->ndim-3] == in->shape[in->ndim-3], BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( out->shape[out->ndim-2] == 1,                     BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( out->shape[out->ndim-1] == in->shape[in->ndim-1], BF_STATUS_INVALID_SHAPE);
	
	// TODO: BF_ASSERT(...);
	BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
	BF_TRY_RETURN(plan->execute(in, out));
}

BFstatus bfBeamformerDestroy(BFbeamformer plan) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	delete plan;
	return BF_STATUS_SUCCESS;
}
