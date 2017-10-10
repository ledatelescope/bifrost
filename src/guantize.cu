/*
 * Copyright (c) 2017, The Bifrost Authors. All rights reserved.
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "assert.hpp"
#include "cuda.hpp"
#include "utils.hu"

#include <limits>
#include <cmath>

// HACK TESTING
#include <iostream>
using std::cout;
using std::endl;

using std::max;
using std::min;

template<typename I, typename F>
inline __device__ F clip(F x) {
	return min_gpu(max_gpu(x,F(minval_gpu<I>())),F(maxval_gpu<I>()));
}

template<typename F>
inline __device__ F clip_4bit(F x) {
	return min_gpu(max_gpu(x,F(-7)),F(7));
}

template<typename F>
inline __device__ F clip_2bit(F x) {
	return min_gpu(max_gpu(x,F(-1)),F(1));
}

template<typename F>
inline __device__ F clip_1bit(F x) {
	return x >= F(0) ? F(1) : F(0);
}

template<typename IType, typename SType, typename OType>
__device__
void guantize(IType ival, SType scale, OType& oval) {
	oval = OType(rint(clip<OType>(ival*scale)));
}

template<typename IType, typename SType, typename OType>
struct GuantizeFunctor {
	SType scale;
	bool  byteswap_in;
	bool  byteswap_out;
	GuantizeFunctor(SType scale_, bool byteswap_in_, bool byteswap_out_)
		: scale(scale_),
		  byteswap_in(byteswap_in_),
		  byteswap_out(byteswap_out_) {}
	__device__ void operator()(IType ival, OType& oval) const {
		if( byteswap_in ) {
			byteswap_gpu(ival, &ival);
		}
		guantize(ival, scale, oval);
		if( byteswap_out ) {
			byteswap_gpu(oval, &oval);
		}
	}
};

template<typename T, typename U, typename Func, typename Size>
__global__
void foreach_simple_gpu(T const* in,
                        U*       out,
                        Size     nelement,
                        Func     func) {
	Size v0 = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x;
	
	if( v0 < nelement ) {
		func(in[v0], out[v0]);

	}
}

template<typename T, typename U, typename Func, typename Size>
inline void launch_foreach_simple_gpu(T const*     in,
                                      U*           out,
                                      Size         nelement,
                                      Func         func,
                                      cudaStream_t stream=0) {
	dim3 block(512, 1); // TODO: Tune this
	Size first = std::min((nelement-1)/block.x+1, 65535ul);
	Size secnd = std::min((nelement - first*block.x) / first + 1, 65535ul);
	if( block.x*first > nelement ) {
		secnd = 1;
	}
	
	dim3 grid(first, secnd);
	/*
	cout << "  Block size is " << block.x << " by " << block.y << endl;
	cout << "  Grid  size is " << grid.x << " by " << grid.y << endl;
	cout << "  Maximum size is " << block.x*grid.x*grid.y << endl;
	if( block.x*grid.x*grid.y >= nelement ) {
		cout << "  -> Valid" << endl;
	}
	*/
	
	void* args[] = {&in,
	                &out, 
	                &nelement, 
	                &func};
	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)foreach_simple_gpu<T,U,Func,Size>,
	                                         grid, block,
	                                         &args[0], 0, stream), 
	                        BF_STATUS_INTERNAL_ERROR);
}

template<typename T, typename Func, typename Size>
__global__
void foreach_simple_gpu_4bit(T const* in,
                             int8_t*  out,
                             Size     nelement,
                             Func     func) {
	Size v0 = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x;
	
	T tempR;
	T tempI;
	int8_t tempO;
	if( v0 < nelement ) {
		tempR = in[2*v0+0];
		tempI = in[2*v0+1];
		if(func.byteswap_in) {
			byteswap_gpu(tempR, &tempR);
			byteswap_gpu(tempI, &tempI);
		}
		tempO = (((int8_t(rint(clip_4bit(tempR*func.scale)))*16)     ) & 0xF0) | \
			   (((int8_t(rint(clip_4bit(tempI*func.scale)))*16) >> 4) & 0x0F);
		if(func.byteswap_out) {
			byteswap_gpu(tempO, &tempO);
		}
		out[v0] = tempO;
	}
}

template<typename T, typename Func, typename Size>
inline void launch_foreach_simple_gpu_4bit(T const*     in,
                                           int8_t*      out,
                                           Size         nelement,
                                           Func         func,
                                           cudaStream_t stream=0) {
	nelement /= 2;
	dim3 block(512, 1); // TODO: Tune this
	Size first = std::min((nelement-1)/block.x+1, 65535ul);
	Size secnd = std::min((nelement - first*block.x) / first + 1, 65535ul);
	if( block.x*first > nelement ) {
		secnd = 1;
	}
	
	dim3 grid(first, secnd);
	/*
	cout << "  Block size is " << block.x << " by " << block.y << endl;
	cout << "  Grid  size is " << grid.x << " by " << grid.y << endl;
	cout << "  Maximum size is " << block.x*grid.x*grid.y << endl;
	if( block.x*grid.x*grid.y >= nelement ) {
		cout << "  -> Valid" << endl;
	}
	*/
	
	void* args[] = {&in,
	                &out, 
	                &nelement, 
	                &func};
	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)foreach_simple_gpu_4bit<T,Func,Size>,
	                                         grid, block,
	                                         &args[0], 0, stream),
	                        BF_STATUS_INTERNAL_ERROR);
}

template<typename T, typename Func, typename Size>
__global__
void foreach_simple_gpu_2bit(T const* in,
                             int8_t*  out,
                             Size     nelement,
                             Func     func) {
	Size v0 = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x;
	
	T tempA;
	T tempB;
	T tempC;
	T tempD;
	int8_t tempO;
	if( v0 < nelement ) {
		tempA = in[4*v0+0];
		tempB = in[4*v0+1];
		tempC = in[4*v0+2];
		tempD = in[4*v0+3];
		if(func.byteswap_in) {
			byteswap_gpu(tempA, &tempA);
			byteswap_gpu(tempB, &tempB);
			byteswap_gpu(tempC, &tempC);
			byteswap_gpu(tempD, &tempD);
		}
		tempO = (((int8_t(rint(clip_2bit(tempA*func.scale)))*64)     ) & 0xC0) | \
			   (((int8_t(rint(clip_2bit(tempB*func.scale)))*64) >> 2) & 0x30) | \
			   (((int8_t(rint(clip_2bit(tempC*func.scale)))*64) >> 4) & 0x0C) | \
			   (((int8_t(rint(clip_2bit(tempD*func.scale)))*64) >> 6) & 0x03);
		if(func.byteswap_out) {
			byteswap_gpu(tempO, &tempO);
		}
		out[v0] = tempO;
	}
}

template<typename T, typename Func, typename Size>
inline void launch_foreach_simple_gpu_2bit(T const*     in,
                                           int8_t*      out,
                                           Size         nelement,
                                           Func         func,
                                           cudaStream_t stream=0) {
	nelement /= 4;
	dim3 block(512, 1); // TODO: Tune this
	Size first = std::min((nelement-1)/block.x+1, 65535ul);
	Size secnd = std::min((nelement - first*block.x) / first + 1, 65535ul);
	if( block.x*first > nelement ) {
		secnd = 1;
	}
	
	dim3 grid(first, secnd);
	/*
	cout << "  Block size is " << block.x << " by " << block.y << endl;
	cout << "  Grid  size is " << grid.x << " by " << grid.y << endl;
	cout << "  Maximum size is " << block.x*grid.x*grid.y << endl;
	if( block.x*grid.x*grid.y >= nelement ) {
		cout << "  -> Valid" << endl;
	}
	*/
	
	void* args[] = {&in,
	                &out, 
	                &nelement, 
	                &func};
	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)foreach_simple_gpu_2bit<T,Func,Size>,
	                                         grid, block,
	                                         &args[0], 0, stream),
	                        BF_STATUS_INTERNAL_ERROR);
}

template<typename T, typename Func, typename Size>
__global__
void foreach_simple_gpu_1bit(T const* in,
                             int8_t*  out,
                             Size     nelement,
                             Func     func) {
	Size v0 = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x;
	
	T tempA;
	T tempB;
	T tempC;
	T tempD;
	T tempE;
	T tempF;
	T tempG;
	T tempH;
	int8_t tempO;
	if( v0 < nelement ) {
		tempA = in[8*v0+0];
		tempB = in[8*v0+1];
		tempC = in[8*v0+2];
		tempD = in[8*v0+3];
		tempE = in[8*v0+4];
		tempF = in[8*v0+5];
		tempG = in[8*v0+6];
		tempH = in[8*v0+7];
		if(func.byteswap_in) {
			byteswap_gpu(tempA, &tempA);
			byteswap_gpu(tempB, &tempB);
			byteswap_gpu(tempC, &tempC);
			byteswap_gpu(tempD, &tempD);
			byteswap_gpu(tempE, &tempE);
			byteswap_gpu(tempF, &tempF);
			byteswap_gpu(tempG, &tempG);
			byteswap_gpu(tempH, &tempH);
		}
		tempO = (((int8_t(rint(clip_1bit(tempA*func.scale)))*128)     ) & 0x08) | \
			   (((int8_t(rint(clip_1bit(tempB*func.scale)))*128) >> 1) & 0x04) | \
			   (((int8_t(rint(clip_1bit(tempC*func.scale)))*128) >> 2) & 0x02) | \
			   (((int8_t(rint(clip_1bit(tempD*func.scale)))*128) >> 3) & 0x10) | \
			   (((int8_t(rint(clip_1bit(tempE*func.scale)))*128) >> 4) & 0x08) | \
			   (((int8_t(rint(clip_1bit(tempF*func.scale)))*128) >> 5) & 0x04) | \
			   (((int8_t(rint(clip_1bit(tempG*func.scale)))*128) >> 6) & 0x02) | \
			   (((int8_t(rint(clip_1bit(tempH*func.scale)))*128) >> 7) & 0x01);
		if(func.byteswap_out) {
			byteswap_gpu(tempO, &tempO);
		}
		out[v0] = tempO;
	}
}

template<typename T, typename Func, typename Size>
inline void launch_foreach_simple_gpu_1bit(T const*     in,
                                           int8_t*      out,
                                           Size         nelement,
                                           Func         func,
                                           cudaStream_t stream=0) {
	nelement /= 8;
	dim3 block(512, 1); // TODO: Tune this
	Size first = std::min((nelement-1)/block.x+1, 65535ul);
	Size secnd = std::min((nelement - first*block.x) / first + 1, 65535ul);
	if( block.x*first > nelement ) {
		secnd = 1;
	}
	
	dim3 grid(first, secnd);
	/*
	cout << "  Block size is " << block.x << " by " << block.y << endl;
	cout << "  Grid  size is " << grid.x << " by " << grid.y << endl;
	cout << "  Maximum size is " << block.x*grid.x*grid.y << endl;
	if( block.x*grid.x*grid.y >= nelement ) {
		cout << "  -> Valid" << endl;
	}
	*/
	
	void* args[] = {&in,
	                &out, 
	                &nelement, 
	                &func};
	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)foreach_simple_gpu_1bit<T,Func,Size>,
	                                         grid, block,
	                                         &args[0], 0, stream),
	                        BF_STATUS_INTERNAL_ERROR);
}

// Instantiation - gunatize functors used in quantize.cpp
//// unsigned
template class GuantizeFunctor<float,float,uint8_t>;
template class GuantizeFunctor<float,double,uint8_t>;
template class GuantizeFunctor<float,float,uint16_t>;
template class GuantizeFunctor<float,double,uint16_t>;
template class GuantizeFunctor<float,float,uint32_t>;
template class GuantizeFunctor<float,double,uint32_t>;
//// signed
template class GuantizeFunctor<float,float,int8_t>;
template class GuantizeFunctor<float,double,int8_t>;
template class GuantizeFunctor<float,float,int16_t>;
template class GuantizeFunctor<float,double,int16_t>;
template class GuantizeFunctor<float,float,int32_t>;
template class GuantizeFunctor<float,double,int32_t>;

// Instantiation - launch_foreach_simple_gpu_1bit calls used in quantize.cpp
template void launch_foreach_simple_gpu_1bit<float,GuantizeFunctor<float,float,uint8_t>,size_t>(float const* in,
                                                                                                int8_t*      out,
                                                                                                size_t       nelement,
                                                                                                GuantizeFunctor<float,float,uint8_t> func,
                                                                                                cudaStream_t stream);
template void launch_foreach_simple_gpu_1bit<float,GuantizeFunctor<float,double,uint8_t>,size_t>(float const* in,
                                                                                                 int8_t*      out,
                                                                                                 size_t       nelement,
                                                                                                 GuantizeFunctor<float,double,uint8_t> func,
                                                                                                 cudaStream_t stream);

// Instantiation - launch_foreach_simple_gpu_2bit calls used in quantize.cpp
template void launch_foreach_simple_gpu_2bit<float,GuantizeFunctor<float,float,uint8_t>,size_t>(float const* in,
                                                                                                int8_t*      out,
                                                                                                size_t       nelement,
                                                                                                GuantizeFunctor<float,float,uint8_t> func,
                                                                                                cudaStream_t stream);
template void launch_foreach_simple_gpu_2bit<float,GuantizeFunctor<float,double,uint8_t>,size_t>(float const* in,
                                                                                                 int8_t*      out,
                                                                                                 size_t       nelement,
                                                                                                 GuantizeFunctor<float,double,uint8_t> func,
                                                                                                 cudaStream_t stream);

// Instantiation - launch_foreach_simple_gpu_4bit calls used in quantize.cpp
template void launch_foreach_simple_gpu_4bit<float,GuantizeFunctor<float,float,uint8_t>,size_t>(float const* in,
                                                                                                int8_t*      out,
                                                                                                size_t       nelement,
                                                                                                GuantizeFunctor<float,float,uint8_t> func,
                                                                                                cudaStream_t stream);
template void launch_foreach_simple_gpu_4bit<float,GuantizeFunctor<float,double,uint8_t>,size_t>(float const* in,
                                                                                                 int8_t*      out,
                                                                                                 size_t       nelement,
                                                                                                 GuantizeFunctor<float,double,uint8_t> func,
                                                                                                 cudaStream_t stream);

// Instantiation - launch_foreach_simple_gpu calls used in quantize.cpp
//// unsigned
template void launch_foreach_simple_gpu<float,uint8_t,GuantizeFunctor<float,float,uint8_t>,size_t>(float const* in,
                                                                                                   uint8_t*     out,
                                                                                                   size_t       nelement,
                                                                                                   GuantizeFunctor<float,float,uint8_t> func,
                                                                                                   cudaStream_t stream);
template void launch_foreach_simple_gpu<float,uint8_t,GuantizeFunctor<float,double,uint8_t>,size_t>(float const* in,
                                                                                                    uint8_t*     out,
                                                                                                    size_t       nelement,
                                                                                                    GuantizeFunctor<float,double,uint8_t> func,
                                                                                                    cudaStream_t stream);
template void launch_foreach_simple_gpu<float,uint16_t,GuantizeFunctor<float,float,uint16_t>,size_t>(float const* in,
                                                                                                     uint16_t*    out,
                                                                                                     size_t       nelement,
                                                                                                     GuantizeFunctor<float,float,uint16_t> func,
                                                                                                     cudaStream_t stream);
template void launch_foreach_simple_gpu<float,uint16_t,GuantizeFunctor<float,double,uint16_t>,size_t>(float const* in,
                                                                                                      uint16_t*    out,
                                                                                                      size_t       nelement,
                                                                                                      GuantizeFunctor<float,double,uint16_t> func,
                                                                                                      cudaStream_t stream);
template void launch_foreach_simple_gpu<float,uint32_t,GuantizeFunctor<float,float,uint32_t>,size_t>(float const* in,
                                                                                                     uint32_t*    out,
                                                                                                     size_t       nelement,
                                                                                                     GuantizeFunctor<float,float,uint32_t> func,
                                                                                                     cudaStream_t stream);
template void launch_foreach_simple_gpu<float,uint32_t,GuantizeFunctor<float,double,uint32_t>,size_t>(float const* in,
                                                                                                      uint32_t*    out,
                                                                                                      size_t       nelement,
                                                                                                      GuantizeFunctor<float,double,uint32_t> func,
                                                                                                      cudaStream_t stream);
//// signed
template void launch_foreach_simple_gpu<float,int8_t,GuantizeFunctor<float,float,int8_t>,size_t>(float const* in,
                                                                                                 int8_t*      out,
                                                                                                 size_t       nelement,
                                                                                                 GuantizeFunctor<float,float,int8_t> func,
                                                                                                 cudaStream_t stream);
template void launch_foreach_simple_gpu<float,int8_t,GuantizeFunctor<float,double,int8_t>,size_t>(float const* in,
                                                                                                  int8_t*      out,
                                                                                                  size_t       nelement,
                                                                                                  GuantizeFunctor<float,double,int8_t> func,
                                                                                                  cudaStream_t stream);
template void launch_foreach_simple_gpu<float,int16_t,GuantizeFunctor<float,float,int16_t>,size_t>(float const* in,
                                                                                                   int16_t*     out,
                                                                                                   size_t       nelement,
                                                                                                   GuantizeFunctor<float,float,int16_t> func,
                                                                                                   cudaStream_t stream);
template void launch_foreach_simple_gpu<float,int16_t,GuantizeFunctor<float,double,int16_t>,size_t>(float const* in,
                                                                                                    int16_t*     out,
                                                                                                    size_t       nelement,
                                                                                                    GuantizeFunctor<float,double,int16_t> func,
                                                                                                    cudaStream_t stream);
template void launch_foreach_simple_gpu<float,int32_t,GuantizeFunctor<float,float,int32_t>,size_t>(float const* in,
                                                                                                   int32_t*     out,
                                                                                                   size_t       nelement,
                                                                                                   GuantizeFunctor<float,float,int32_t> func,
                                                                                                   cudaStream_t stream);
template void launch_foreach_simple_gpu<float,int32_t,GuantizeFunctor<float,double,int32_t>,size_t>(float const* in,
                                                                                                    int32_t*     out,
                                                                                                    size_t       nelement,
                                                                                                    GuantizeFunctor<float,double,int32_t> func,
                                                                                                    cudaStream_t stream);

