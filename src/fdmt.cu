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

#include <bifrost/fdmt.h>
#include "assert.hpp"
#include "utils.hpp"
#include "workspace.hpp"
#include "cuda.hpp"
#include "trace.hpp"

//#include <limits>

#include <math_constants.h> // For CUDART_NAN_F
#include <thrust/device_vector.h>

#include <vector>
#include <map>
#include <string>
#include <complex>

// HACK TESTING
#include <iostream>
using std::cout;
using std::endl;

// Note: Can be tuned over block shape
template<typename InType, typename OutType>
__global__
void fdmt_init_kernel(int                         ntime,
                      int                         nchan,
                      int                         nbatch,
                      bool                        reverse_band,
                      bool                        reverse_time,
                      int     const* __restrict__ d_offsets,
                      InType  /*const* __restrict__*/ d_in,
                      int                         istride,
                      int                         ibatchstride,
                      OutType*       __restrict__ d_out,
                      int                         ostride,
                      int                         obatchstride) {
	int t0 = threadIdx.x + blockIdx.x*blockDim.x;
	int c0 = threadIdx.y + blockIdx.y*blockDim.y;
	int b0 = blockIdx.z;
	for( int b=b0; b<nbatch; b+=gridDim.z ) {
	for( int c=c0; c<nchan; c+=blockDim.y*gridDim.y ) {
		int offset = d_offsets[c];
		int ndelay = d_offsets[c+1] - offset;
		for( int t=t0; t<ntime; t+=blockDim.x*gridDim.x ) {
			OutType tmp(0);
			for( int d=0; d<ndelay; ++d ) {
				// Note: This fills the unused elements with NaNs
				OutType outval(CUDART_NAN_F);//std::numeric_limits<OutType>::quiet_NaN());
				if( t >= d ) {
					int c_ = reverse_band ? nchan-1 - c : c;
					int t_ = reverse_time ? ntime-1 - t : t;
					tmp += d_in[(t_-d) + istride*c_ + ibatchstride*b];
					// TODO: Check effect of not-/using sqrt
					//         The final paper has no sqrt (i.e., computation is just the mean)
					//outval = tmp * rsqrtf(d+1);
					outval = tmp * (1.f/(d+1));
				}
				d_out[t + ostride*(offset+d) + obatchstride*b] = outval;
			}
		}
	}
	}
}

// Note: Can be tuned over block shape
template<typename DType>
__global__
void fdmt_exec_kernel(int                       ntime,
                      int                       nrow,
                      int                       nbatch,
                      bool                      is_final_step,
                      bool                      reverse_time,
                      int   const* __restrict__ d_delays,
                      int2  const* __restrict__ d_srcrows,
                      DType const* __restrict__ d_in,
                      int                       istride,
                      int                       ibatchstride,
                      DType*       __restrict__ d_out,
                      int                       ostride,
                      int                       obatchstride) {
	int t0 = threadIdx.x + blockIdx.x*blockDim.x;
	int r0 = threadIdx.y + blockIdx.y*blockDim.y;
	int b0 = blockIdx.z;
	for( int b=b0; b<nbatch; b+=gridDim.z ) {
	for( int r=r0; r<nrow; r+=blockDim.y*gridDim.y ) {
		int delay   = d_delays[r];
		int srcrow0 = d_srcrows[r].x;
		int srcrow1 = d_srcrows[r].y;
		for( int t=t0; t<ntime; t+=blockDim.x*gridDim.x ) {
			// Avoid elements that go unused due to diagonal reindexing
			if( is_final_step && t < r ) {
				//int ostride_ = ostride - reverse_time;
				//d_out[t + ostride_*r] = CUDART_NAN_F;
				continue;
			}
			// HACK TESTING
			////if( ostride < ntime && t >= ntime-1 - r ) {
			//if( ostride != ntime && t < r ) {
			//	int ostride_ = ostride - (ostride > ntime);
			//	d_out[t + ostride_*r] = CUDART_NAN_F;
			//	continue;
			//}// else if( ostride > ntime && t >= ntime - r ) {
				//	//d_out[t - (ntime-1) + ostride*r] = CUDART_NAN_F;
					//	continue;
				//}
			
			// Note: Non-existent rows are signified by -1
			//if( t == 0 && r == 0 ) {
			//	printf("t,srcrow0,srcrow1,istride = %i, %i, %i, %i\n", t, srcrow0, srcrow1, istride);
			//}
			//if( threadIdx.x == 63 && blockIdx.y == 4 ) {
			//printf("istride = %i, srcrow0 = %i, srcrow1 = %i, d_in = %p\n", istride, srcrow0, srcrow1, d_in);
				//}
			//if( t == 0 ) {// && r == 1 ) {
			//	printf("istride = %i, srcrow0 = %i, srcrow1 = %i, d_in = %p\n", istride, srcrow0, srcrow1, d_in);
			//}
			DType outval = (srcrow0 != -1) ? d_in[ t        + istride*srcrow0 + ibatchstride*b] : 0;
			if( t >= delay ) {
				outval  += (srcrow1 != -1) ? d_in[(t-delay) + istride*srcrow1 + ibatchstride*b] : 0;
			}
			int t_ = (is_final_step && reverse_time) ? ntime-1 - t : t;
			d_out[t_ + ostride*r + obatchstride*b] = outval;
		}
	}
	}
}

template<typename InType, typename OutType>
inline
void launch_fdmt_init_kernel(int            ntime,
                             int            nchan,
                             int            nbatch,
                             bool           reverse_band,
                             bool           reverse_time,
                             //int     const* d_ndelays,
                             int     const* d_offsets,
                             InType  /*const**/ d_in,
                             int            istride,
                             int            ibatchstride,
                             OutType*       d_out,
                             int            ostride,
                             int            obatchstride,
                             cudaStream_t   stream=0) {
	dim3 block(256, 1); // TODO: Tune this
	dim3 grid(std::min((ntime-1)/block.x+1, 65535u),
	          std::min((nchan-1)/block.y+1, 65535u));
	void* args[] = {&ntime,
	                &nchan,
	                &nbatch,
	                &reverse_band,
	                &reverse_time,
	                &d_offsets,
	                &d_in,
	                &istride,
	                &ibatchstride,
	                &d_out,
	                &ostride,
	                &obatchstride};
	BF_CHECK_CUDA_EXCEPTION(
		cudaLaunchKernel((void*)fdmt_init_kernel<InType,OutType>,
		                 grid, block,
		                 &args[0], 0, stream),
		BF_STATUS_INTERNAL_ERROR);
}

template<typename DType>
inline
void launch_fdmt_exec_kernel(int          ntime,
                             int          nrow,
                             int          nbatch,
                             bool         is_final_step,
                             bool         reverse_time,
                             int   const* d_delays,
                             int2  const* d_srcrows,
                             DType const* d_in,
                             int          istride,
                             int          ibatchstride,
                             DType*       d_out,
                             int          ostride,
                             int          obatchstride,
                             cudaStream_t stream=0) {
	//cout << "LAUNCH " << d_in << ", " << d_out << endl;
	dim3 block(256, 1); // TODO: Tune this
	dim3 grid(std::min((ntime-1)/block.x+1, 65535u),
	          std::min((nrow -1)/block.y+1, 65535u));
	void* args[] = {&ntime,
	                &nrow,
	                &nbatch,
	                &is_final_step,
	                &reverse_time,
	                &d_delays,
	                &d_srcrows,
	                &d_in,
	                &istride,
	                &ibatchstride,
	                &d_out,
	                &ostride,
	                &obatchstride};
	BF_CHECK_CUDA_EXCEPTION(
		cudaLaunchKernel((void*)fdmt_exec_kernel<DType>,
		                 grid, block,
		                 &args[0], 0, stream),
		BF_STATUS_INTERNAL_ERROR);
}
/*
**** 4096
**** 4096
**** 2048
**** 1066
**** 650
**** 475
**** 381
**** 337
**** 316
**** 302
**** 299
**** 295
**** 293
SB 3
delay 135
Step 10 prev: 58, 78
       srcs: 57, 78
NROW_MAX = 4096
STEP 1
STEP 2
STEP 3
STEP 4
STEP 5
STEP 6
STEP 7
STEP 8
STEP 9
STEP 10
STEP 11
*/

class BFfdmt_impl {
	typedef int    IType;
	typedef double FType;
	typedef int2   IndexPair;
public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
	typedef float  DType;
private:
	IType _nchan;
	IType _max_delay;
	FType _f0;
	FType _df;
	FType _exponent;
	IType _nrow_max;
	IType _plan_stride;
	IType _buffer_stride;
	IType _batch_stride;
	std::vector<IType>                   _offsets;
	std::vector<std::vector<IndexPair> > _step_srcrows;
	std::vector<std::vector<IType> >     _step_delays;
	IType*     _d_offsets;
	IndexPair* _d_step_srcrows;
	IType*     _d_step_delays;
	DType*     _d_buffer_a;
	DType*     _d_buffer_b;
	Workspace _plan_storage;
	Workspace _exec_storage;
	// TODO: Use something other than Thrust
	thrust::device_vector<char> _dv_plan_storage;
	thrust::device_vector<char> _dv_exec_storage;
	cudaStream_t _stream;
	bool _reverse_band;
	
	FType cfreq(IType chan) {
		return _f0 + _df*chan;
	}
	FType rel_delay(FType flo, FType fhi, FType fmin, FType fmax) {
		FType g = _exponent;
		// Note: We use complex math in order to support negative frequencies
		//         (the result is real regardless).
		std::complex<FType> c_flo=flo, c_fhi=fhi, c_fmin=fmin, c_fmax=fmax;
		std::complex<FType> numer = std::pow(c_flo,  g) - std::pow(c_fhi,  g);
		std::complex<FType> denom = std::pow(c_fmin, g) - std::pow(c_fmax, g);
		FType eps = std::numeric_limits<FType>::epsilon();
		if( std::norm(denom) < eps*eps ) {
			// Note: The only time I've seen this fail is when nchan==1
			BF_ASSERT_EXCEPTION(std::norm(numer) < eps*eps,
			                    BF_STATUS_INTERNAL_ERROR);
			return 0;
		}
		std::complex<FType> result = numer / denom;
		BF_ASSERT_EXCEPTION(std::abs(result.imag()) <= eps,
		                    BF_STATUS_INTERNAL_ERROR);
		return result.real();
	}
	FType rel_delay(FType flo, FType fhi) {
		FType fmin = cfreq(0);
		FType fmax = cfreq(_nchan-1);
		//std::swap(fmin, fmax);
		//FType fmax = cfreq(_nchan); // HACK TESTING
		return rel_delay(flo, fhi, fmin, fmax);
	}
	IType subband_ndelay(FType f0, FType df) {
		FType fracdelay = rel_delay(f0, f0+df);
		FType fmaxdelay = fracdelay*(_max_delay-1);
		IType ndelay = IType(::ceil(fmaxdelay)) + 1;
		return ndelay;
	}
public:
	BFfdmt_impl() : _nchan(0), _max_delay(0), _f0(0), _df(0), _exponent(0),
	                _stream(g_cuda_stream) {}
	inline IType nchan()     const { return _nchan; }
	inline IType max_delay() const { return _max_delay; }
	void init(IType nchan,
	          IType max_delay,
	          FType f0,
	          FType df,
	          FType exponent) {
		BF_TRACE();
		if( df < 0. ) {
			_reverse_band = true;
			f0 += (nchan-1)*df;
			df *= -1;
		} else {
			_reverse_band = false;
		}
		if( nchan     == _nchan     &&
		    max_delay == _max_delay &&
		    f0        == _f0        &&
		    df        == _df        &&
		    exponent  == _exponent ) {
			return;
		}
		_f0        = f0;
		_df        = df;
		_nchan     = nchan;
		_max_delay = max_delay;
		_exponent  = exponent;
		// Note: Initialized with 1 entry as dummy for initialization step
		std::vector<std::vector<IndexPair> > step_subband_parents(1);
		IType nsubband = _nchan;
		while( nsubband > 1 ) {
			IType step = step_subband_parents.size();
			step_subband_parents.push_back(std::vector<IndexPair>());
			for( IType sb=0; sb<nsubband; sb+=2 ) {
				IType parent0 = sb;
				IType parent1 = sb+1;
				if( nsubband % 2 ) {
					// Note: Alternating left/right-biased merging scheme
					if( (step-1) % 2 ) {
						parent0 -= 1; // Note: First entry becomes -1 => non-existent
						parent1 -= 1;
					} else {
						// Note: Last entry becomes -1 => non-existent
						if( parent1 == nsubband ) parent1 = -1;
					}
				}
				//cout << step << ": " << parent0 << ", " << parent1 << endl;
				IndexPair parents = make_int2(parent0, parent1);
				step_subband_parents[step].push_back(parents);
			}
			nsubband = step_subband_parents[step].size();
		}
		// Note: Includes initialization step
		IType nstep = step_subband_parents.size();
		
		std::vector<std::vector<IType> > step_subband_nchans(nstep);
		step_subband_nchans[0].assign(_nchan, 1);
		for( IType step=1; step<nstep; ++step ) {
			IType nsubband = step_subband_parents[step].size();
			step_subband_nchans[step].resize(nsubband);
			for( IType sb=0; sb<nsubband; ++sb ) {
				IndexPair parents = step_subband_parents[step][sb];
				IType p0 = parents.x;//first;
				IType p1 = parents.y;//second;
				IType parent0_nchan = (p0!=-1) ? step_subband_nchans[step-1][p0] : 0;
				IType parent1_nchan = (p1!=-1) ? step_subband_nchans[step-1][p1] : 0;
				IType child_nchan = parent0_nchan + parent1_nchan;
				step_subband_nchans[step][sb] = child_nchan;
			}
		}
		
		std::vector<std::vector<IType> > step_subband_chan_offsets(nstep);
		std::vector<std::vector<IType> > step_subband_row_offsets(nstep);
		IType nrow_max = 0;
		for( IType step=0; step<nstep; ++step ) {
			IType nsubband = step_subband_nchans[step].size();
			// Note: +1 to store the total in the last element
			//        (The array will hold a complete exclusive scan)
			step_subband_chan_offsets[step].resize(nsubband+1);
			step_subband_row_offsets[step].resize(nsubband+1);
			IType chan0 = 0;
			IType row_offset = 0;
			for( IType sb=0; sb<nsubband; ++sb ) {
				IType nchan = step_subband_nchans[step][sb];
				FType f0 = cfreq(chan0) - (step == 0 ? 0.5*_df : 0.);
				//FType f0 = cfreq(chan0); // HACK TESTING
				FType df = _df * (step == 0 ? 1 : nchan-1);
				//FType df = _df * nchan; // HACK TESTING
				//cout << "df = " << df << endl;
				IType ndelay = subband_ndelay(f0, df);
				//cout << "NDELAY = " << ndelay << endl;
				step_subband_chan_offsets[step][sb] = chan0;
				step_subband_row_offsets[step][sb] = row_offset;
				chan0 += nchan;
				row_offset += ndelay;
			}
			step_subband_chan_offsets[step][nsubband] = chan0;
			step_subband_row_offsets[step][nsubband] = row_offset;
			nrow_max = std::max(nrow_max, row_offset);
			//*cout << "**** Nrow: " << row_offset << endl;
		}
		// Save for use during initialization
		//plan->_init_subband_row_offsets = step_subband_row_offsets[0];
		_offsets = step_subband_row_offsets[0];
		_nrow_max = nrow_max;
		//cout << "**** " << _nrow_max << endl;
		
		// Note: First entry in these remains empty
		std::vector<std::vector<IndexPair> > step_srcrows(nstep);
		std::vector<std::vector<IType> >     step_delays(nstep);
		for( IType step=1; step<nstep; ++step ) {
			IType nsubband = step_subband_nchans[step].size();
			IType nrow     = step_subband_row_offsets[step][nsubband];
			//*cout << "nrow " << nrow << endl;
			step_srcrows[step].resize(nrow);
			step_delays[step].resize(nrow);
			for( IType sb=0; sb<nsubband; ++sb ) {
				IndexPair parents = step_subband_parents[step][sb];
				IType p0 = parents.x;//first;
				IType p1 = parents.y;//second;
				// TODO: Setting these to 1 instead of 0 in the exceptional case fixed some indexing
				//         issues, but should double-check that the results are good.
				IType p0_nchan = (p0!=-1) ? step_subband_nchans[step-1][p0] : 1;
				IType p1_nchan = (p1!=-1) ? step_subband_nchans[step-1][p1] : 1;
				// Note: If first parent doesn't exist, then it effectively starts where the second parent starts
				//       If second parent doesn't exist, then it effectively starts where the first parent ends
				IType p0_chan0 = step_subband_chan_offsets[step-1][(p0!=-1) ? p0 : p1];
				IType p1_chan0 = step_subband_chan_offsets[step-1][(p1!=-1) ? p1 : p0];
				if( p1 == -1 ) {
					p1_chan0 += (p0_nchan-1);
				}
				FType flo    = cfreq(p0_chan0);
				FType fmidlo = cfreq(p0_chan0 + (p0_nchan-1));
				FType fmidhi = cfreq(p1_chan0);
				FType fhi    = cfreq(p1_chan0 + (p1_nchan-1));
				FType cmidlo = rel_delay(flo, fmidlo, flo, fhi);
				FType cmidhi = rel_delay(flo, fmidhi, flo, fhi);
				/*
				// HACK TESTING
				FType flo    = cfreq(p0_chan0) - 0.5*_df;
				FType fmidlo = flo + (p0_nchan-1)*_df;
				FType fmidhi = flo + p0_nchan*_df;
				FType fhi    = flo + (p0_nchan + p1_nchan - 1)*_df;
				FType cmidlo = rel_delay(fmidlo, flo, fhi, flo);
				FType cmidhi = rel_delay(fmidhi, flo, fhi, flo);
				*/
				//cout << p0 << ", " << p1 << endl;
				//cout << p0_chan0 << ", " << p0_nchan << "; " << p1_chan0 << ", " << p1_nchan << endl;
				//cout << cmidlo << ", " << cmidhi << endl;
				
				// TODO: See if should use same approach with these as in fdmt.py
				IType beg = step_subband_row_offsets[step][sb];
				IType end = step_subband_row_offsets[step][sb+1];
				IType ndelay = end - beg;
				for( IType delay=0; delay<ndelay; ++delay ) {
					IType dmidlo = (IType)::round(delay*cmidlo);
					IType dmidhi = (IType)::round(delay*cmidhi);
					IType drest = delay - dmidhi;
					assert( dmidlo <= delay );
					assert( dmidhi <= delay );
					IType prev_beg  = (p0!=-1) ? step_subband_row_offsets[step-1][p0]   : -1;
					IType prev_mid0 = (p0!=-1) ? step_subband_row_offsets[step-1][p0+1] : -1;
					IType prev_mid1 = (p1!=-1) ? step_subband_row_offsets[step-1][p1]   : -1;
					IType prev_end  = (p1!=-1) ? step_subband_row_offsets[step-1][p1+1] : -1;
					// HACK WAR for strange indexing error observed only when nchan=4096
					if( p1 != -1 && drest >= prev_end - prev_mid1 ) {
						drest -= 1;
					}
					if( (p0 != -1 && dmidlo >= prev_mid0 - prev_beg) ||
					    (p1 != -1 && drest  >= prev_end - prev_mid1) ) {
						cout << "FDMT DEBUGGING INFO" << endl;
						cout << "SB " << sb << endl;
						cout << "delay " << delay << endl;
						cout << "Step " << step << " prev: " << prev_mid0 - prev_beg << ", " << prev_end - prev_mid1 << endl;
						cout << "       srcs: " << dmidlo << ", " << drest << endl;
						
					}
					assert( p0 == -1 || dmidlo < prev_mid0 - prev_beg );
					assert( p1 == -1 || drest  < prev_end - prev_mid1 );
					IType dst_row  = step_subband_row_offsets[step  ][sb] + delay;
					IType src_row0 = (p0!=-1) ? step_subband_row_offsets[step-1][p0] + dmidlo : -1;
					IType src_row1 = (p1!=-1) ? step_subband_row_offsets[step-1][p1] + drest  : -1;
					step_srcrows[step][dst_row].x = src_row0;//first  = src_row0;
					//cout << "step " << step << ", dst_row = " << dst_row << ", delay = " << dmidhi << ", src_row0 = " << src_row0 << ", src_row1 = " << src_row1 << endl;
					step_srcrows[step][dst_row].y = src_row1;//second = src_row1;
					step_delays[step][dst_row] = dmidhi;
					//IType prev_nsubband = step_subband_nchans[step-1].size();
					//IType prev_nrow = step_subband_row_offsets[step-1][prev_nsubband];
				}
			}
		}
		// Save for use during execution
		_step_srcrows = step_srcrows;
		_step_delays  = step_delays;
	}
	bool init_plan_storage(void* storage_ptr, BFsize* storage_size) {
		BF_TRACE();
		BF_TRACE_STREAM(_stream);
		enum {
			ALIGNMENT_BYTES = 512,
			ALIGNMENT_ELMTS = ALIGNMENT_BYTES / sizeof(int)
		};
		Workspace workspace(ALIGNMENT_BYTES);
		_plan_stride = round_up(_nrow_max, ALIGNMENT_ELMTS);
		//int nstep_execute = _step_delays.size() - 1;
		int nstep = _step_delays.size();
		workspace.reserve(_nchan+1, &_d_offsets);
		workspace.reserve(nstep*_plan_stride, &_d_step_srcrows);
		workspace.reserve(nstep*_plan_stride, &_d_step_delays);
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
		//std::cout << "workspace.size() = " << workspace.size() << std::endl;
		//_d_offsets = (IType*)0x123;
		//std::cout << "_d_offsets = " << _d_offsets << std::endl;
		//std::cout << "storage_ptr = " << storage_ptr << std::endl;
		workspace.commit(storage_ptr);
		//std::cout << "_d_offsets = " << _d_offsets << std::endl;
		BF_CHECK_CUDA_EXCEPTION( cudaMemcpyAsync(_d_offsets,
		                                         &_offsets[0],
		                                         sizeof(int )*_offsets.size(),
		                                         cudaMemcpyHostToDevice,
		                                         _stream),
		                         BF_STATUS_MEM_OP_FAILED );
		for( int step=0; step<nstep; ++step ) {
			BF_CHECK_CUDA_EXCEPTION( cudaMemcpyAsync(_d_step_srcrows + step*_plan_stride,
			                                         &_step_srcrows[step][0],
			                                         sizeof(int2)*_step_srcrows[step].size(),
			                                         cudaMemcpyHostToDevice,
			                                         _stream),
			               BF_STATUS_MEM_OP_FAILED );
			BF_CHECK_CUDA_EXCEPTION( cudaMemcpyAsync(_d_step_delays  + step*_plan_stride,
			                                         &_step_delays[step][0],
			                                         sizeof(int)*_step_delays[step].size(),
			                                         cudaMemcpyHostToDevice,
			                                         _stream),
			               BF_STATUS_MEM_OP_FAILED );
		}
		BF_CHECK_CUDA_EXCEPTION( cudaStreamSynchronize(_stream),
		                         BF_STATUS_DEVICE_ERROR );
		return true;
	}
	bool init_exec_storage(void* storage_ptr, BFsize* storage_size,
	                       size_t ntime, size_t nbatch) {
		BF_TRACE();
		enum {
			ALIGNMENT_BYTES = 512,
			ALIGNMENT_ELMTS = ALIGNMENT_BYTES / sizeof(DType)
		};
		Workspace workspace(ALIGNMENT_BYTES);
		//std::cout << "ntime = " << ntime << std::endl;
		//std::cout << "_nrow_max = " << _nrow_max << std::endl;
		_buffer_stride = round_up(ntime, ALIGNMENT_ELMTS);
		_batch_stride  = _nrow_max*_buffer_stride;
		//std::cout << "_buffer_stride = " << _buffer_stride << std::endl;
		// TODO: Check if truly safe to allocate smaller buffer_b
		workspace.reserve(nbatch*_batch_stride, &_d_buffer_a);
		workspace.reserve(nbatch*_batch_stride, &_d_buffer_b);
		if( storage_size ) {
			if( !storage_ptr ) {
				//cout << "++++ returning storage size" << endl;
				// Return required storage size
				*storage_size = workspace.size();
				return false;
			} else {
				//cout << "++++ using user storage" << endl;
				BF_ASSERT_EXCEPTION(*storage_size >= workspace.size(),
				                    BF_STATUS_INSUFFICIENT_STORAGE);
			}
		} else {
			//cout << "++++ auto-allocating storage" << endl;
			// Auto-allocate storage
			BF_ASSERT_EXCEPTION(!storage_ptr, BF_STATUS_INVALID_ARGUMENT);
			_dv_exec_storage.resize(workspace.size());
			storage_ptr = thrust::raw_pointer_cast(&_dv_exec_storage[0]);
			//std::cout << "*** exec storage_ptr = " << storage_ptr << std::endl;
		}
		//cout << "++++ committing" << endl;
		workspace.commit(storage_ptr);
		return true;
	}
	void execute(BFarray const* in,
	             BFarray const* out,
	             size_t         ntime,
	             size_t         nbatch,
	             bool           negative_delays) {
		BF_TRACE();
		BF_TRACE_STREAM(_stream);
		BF_ASSERT_EXCEPTION(out->dtype == BF_DTYPE_F32, BF_STATUS_UNSUPPORTED_DTYPE);
		BF_ASSERT_EXCEPTION(   out->strides[in->ndim-1] == 4, BF_STATUS_UNSUPPORTED_STRIDE);
		int ndim = in->ndim;
		DType* d_ibuf = _d_buffer_b;
		DType* d_obuf = _d_buffer_a;
		BF_ASSERT_EXCEPTION(in->strides[ndim-2] % in->strides[ndim-1] == 0,
		                    BF_STATUS_UNSUPPORTED_STRIDE);
		BF_ASSERT_EXCEPTION(out->strides[ndim-2] % out->strides[ndim-1] == 0,
		                    BF_STATUS_UNSUPPORTED_STRIDE);
		size_t istride =  in->strides[ndim-2] /  in->strides[ndim-1];
		size_t ostride = out->strides[ndim-2] / out->strides[ndim-1];
		BF_ASSERT_EXCEPTION( in->strides[in->ndim-2] > 0, BF_STATUS_UNSUPPORTED_STRIDE);
		BF_ASSERT_EXCEPTION(out->strides[in->ndim-2] > 0, BF_STATUS_UNSUPPORTED_STRIDE);
		
		size_t ibatchstride = 0;
		size_t obatchstride = 0;
		if( in->ndim == 3 ) {
			BF_ASSERT_EXCEPTION(in->strides[ndim-3] % in->strides[ndim-1] == 0,
			                    BF_STATUS_UNSUPPORTED_STRIDE);
			BF_ASSERT_EXCEPTION(out->strides[ndim-3] % out->strides[ndim-1] == 0,
			                    BF_STATUS_UNSUPPORTED_STRIDE);
			ibatchstride =  in->strides[ndim-3] /  in->strides[ndim-1];
			obatchstride = out->strides[ndim-3] / out->strides[ndim-1];
		}
		
		BF_ASSERT_EXCEPTION(in->strides[ndim-1] == BF_DTYPE_NBYTE(in->dtype),
		                    BF_STATUS_UNSUPPORTED_STRIDE);
		
		//bool reverse_time = (in->strides[in->ndim-1] < 0);
		bool reverse_time = negative_delays;
		
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
#define LAUNCH_FDMT_INIT_KERNEL(IterType) \
		launch_fdmt_init_kernel(ntime, _nchan, nbatch, \
		                        _reverse_band, reverse_time, \
		                        _d_offsets, \
		                        (IterType)in->data, istride, ibatchstride, \
		                        d_obuf, _buffer_stride, _batch_stride, \
		                        _stream)
		
		switch( in->dtype ) {
			// HACK testing disabled
			// TODO: Get NbitReader working
			//case BF_DTYPE_I1:  LAUNCH_FDMT_INIT_KERNEL(NbitReader<1>); break;
			//case BF_DTYPE_I2:  LAUNCH_FDMT_INIT_KERNEL(NbitReader<2>); break;
			//case BF_DTYPE_I4:  LAUNCH_FDMT_INIT_KERNEL(NbitReader<4>); break;
		case BF_DTYPE_I8:  LAUNCH_FDMT_INIT_KERNEL(const int8_t*);   break;
		case BF_DTYPE_I16: LAUNCH_FDMT_INIT_KERNEL(const int16_t*);  break;
		case BF_DTYPE_I32: LAUNCH_FDMT_INIT_KERNEL(const int32_t*);  break;
		case BF_DTYPE_U8:  LAUNCH_FDMT_INIT_KERNEL(const uint8_t*);  break;
		case BF_DTYPE_U16: LAUNCH_FDMT_INIT_KERNEL(const uint16_t*); break;
		case BF_DTYPE_U32: LAUNCH_FDMT_INIT_KERNEL(const uint32_t*); break;
		case BF_DTYPE_F32: LAUNCH_FDMT_INIT_KERNEL(const float*);    break;
		default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
		}
#undef LAUNCH_FDMT_INIT_KERNEL
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
		std::swap(d_ibuf, d_obuf);
		
		size_t ostride_cur      = _buffer_stride;
		size_t obatchstride_cur = _batch_stride;
		IType nstep = _step_delays.size();
		for( int step=1; step<nstep; ++step ) {
			//cout << "STEP " << step << endl;
			IType nrow = _step_srcrows[step].size();
			//cout << "nrow " << nrow << endl;
			if( step == nstep-1 ) {
				d_obuf  = (DType*)out->data;
				ostride_cur = ostride;
				// HACK TESTING diagonal reindexing to align output with TOA at highest freq
				ostride_cur += reverse_time ? +1 : -1;
				obatchstride_cur = obatchstride;
			}
			launch_fdmt_exec_kernel(ntime, nrow, nbatch, (step==nstep-1), reverse_time,
			                        _d_step_delays  + step*_plan_stride,
			                        _d_step_srcrows + step*_plan_stride,
			                        d_ibuf, _buffer_stride, _batch_stride,
			                        d_obuf, ostride_cur, obatchstride_cur,
			                        _stream);
			std::swap(d_ibuf, d_obuf);
		}
		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
	}
	void set_stream(cudaStream_t stream) {
		_stream = stream;
	}
};

BFstatus bfFdmtCreate(BFfdmt* plan_ptr) {
	BF_TRACE();
	BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*plan_ptr = new BFfdmt_impl(),
	                   *plan_ptr = 0);
}
// **TODO: Passing 'BFarray const* in' here could replace nchan, f0, df and space if BFarray included dimension scales
//           Also, could potentially set the output dimension scales (dm0, ddm)
//           OR, could just leave these to higher-level wrappers (e.g., Python)
//             This might be for the best in the short term
BFstatus bfFdmtInit(BFfdmt  plan,
                    BFsize  nchan,
                    BFsize  max_delay,
                    double  f0,
                    double  df,
                    double  exponent,
                    BFspace space,
                    void*   plan_storage,
                    BFsize* plan_storage_size) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	// TODO: Is there any sensible/natural way to handle nchan==1?
	BF_ASSERT(nchan > 1, BF_STATUS_INVALID_ARGUMENT);
	BF_ASSERT(space_accessible_from(space, BF_SPACE_CUDA),
	          BF_STATUS_UNSUPPORTED_SPACE);
	BF_TRY(plan->init(nchan, max_delay, f0, df, exponent));
	BF_TRY_RETURN(plan->init_plan_storage(plan_storage, plan_storage_size));
}
BFstatus bfFdmtSetStream(BFfdmt      plan,
                         void const* stream) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}
BFstatus bfFdmtExecute(BFfdmt         plan,
                       BFarray const* in,
                       BFarray const* out,
                       BFbool         negative_delays,
                       void*          exec_storage,
                       BFsize*        exec_storage_size) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
	BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
	BF_ASSERT(in->ndim == out->ndim, BF_STATUS_INVALID_SHAPE);
	BF_ASSERT( in->shape[ in->ndim-2] == plan->nchan(),     BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(out->shape[out->ndim-2] == plan->max_delay(), BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(  in->shape[in->ndim-1] == out->shape[out->ndim-1], BF_STATUS_INVALID_SHAPE);
	// TODO: BF_ASSERT(...);
	int ndim = in->ndim;
	size_t ntime  = in->shape[in->ndim-1];
	size_t nbatch = 1;
	BFarray out_flattened, in_flattened;
	// Handle batch dims
	if( ndim > 2 ) {
		// Keep the last 3 dims but attempt to flatten all others
		unsigned long keep_dims_mask = 0x7 << (ndim-3);
		keep_dims_mask |= padded_dims_mask(out);
		keep_dims_mask |= padded_dims_mask(in);
		flatten(out, &out_flattened, keep_dims_mask);
		flatten(in,   &in_flattened, keep_dims_mask);
		out = &out_flattened;
		in  =  &in_flattened;
		BF_ASSERT(in_flattened.ndim == out_flattened.ndim,
		          BF_STATUS_INTERNAL_ERROR);
		// TODO: Use streams to support multiple non-contiguous batch dims
		//         (Like in linalg.cu)
		BF_ASSERT(in_flattened.ndim == 3, BF_STATUS_UNSUPPORTED_SHAPE);
		BF_ASSERT_EXCEPTION(in_flattened.shape[0] == out_flattened.shape[0],
		                    BF_STATUS_INVALID_SHAPE);
		nbatch = in->shape[0];
	}
	bool ready;
	BF_TRY(ready = plan->init_exec_storage(exec_storage, exec_storage_size,
	                                       ntime, nbatch));
	if( !ready ) {
		// Just requesting exec_storage_size, not ready to execute yet
		return BF_STATUS_SUCCESS;
	}
	BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
	BF_TRY_RETURN(plan->execute(in, out, ntime, nbatch, negative_delays));
}

BFstatus bfFdmtDestroy(BFfdmt plan) {
	BF_TRACE();
	BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
	delete plan;
	return BF_STATUS_SUCCESS;
}
