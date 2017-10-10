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

// HACK TESTING
#include <iostream>
using std::cout;
using std::endl;

// 2x 4-bit --> 2x 8-bit (unsigned)
inline __device__ void gunpack(uint8_t   ival,
                               uint16_t& oval,
                               bool      byte_reverse,
                               bool      align_msb,
                               bool      conjugate) {
	// Note: Ignores conjugate
	if( byte_reverse ) {
		// ........ABCDEFGH
		// EFGH....ABCD....
		oval = ival;
		oval = (oval | (oval << 12)) & 0xF0F0;
	} else {
		// ....ABCDEFGH....
		// ABCD....EFGH....
		oval = ival << 4;
		oval = (oval | (oval <<  4)) & 0xF0F0;
	}
	if( !align_msb ) {
		// >>>>ABCD>>>>EFGH
		oval >>= 4;
	}
}

// 4x 2-bit --> 4x 8-bit (unsigned)
inline __device__ void gunpack(uint8_t   ival,
                               uint32_t& oval,
                               bool      byte_reverse,
                               bool      align_msb,
                               bool      conjugate) {
	// Note: Ignores conjugate
	// ..................ABCDEFGH......
	// ......ABCD............EFGH......
	// AB......CD......EF......GH......
	oval = ival << 6;
	oval = (oval | (oval << 12)) & 0x03C003C0;
	oval = (oval | (oval <<  6)) & 0xC0C0C0C0;
	if( byte_reverse) {
		byteswap_gpu(oval, &oval);
	}
	if( !align_msb ) {
		// >>>>>>AB>>>>>>CD>>>>>>EF>>>>>>GH
		oval >>= 6;
	}
}

// 8x 1-bit --> 8x 8-bit (unsigned)
inline __device__ void gunpack(uint8_t   ival,
                                        uint64_t& oval,
                                        bool      byte_reverse,
                                        bool      align_msb,
                                        bool      conjugate) {
	// Note: Ignores conjugate
	// .................................................ABCDEFGH.......
	// .....................ABCD............................EFGH.......
	// .......AB..............CD..............EF..............GH.......
	// A.......B.......C.......D.......E.......F.......G.......H.......
	oval = ival << 7;
	oval = (oval | (oval << 28)) & 0x0000078000000780;
	oval = (oval | (oval << 14)) & 0x0180018001800180;
	oval = (oval | (oval <<  7)) & 0x8080808080808080;
	if( byte_reverse) {
		byteswap_gpu(oval, &oval);
	}
	if( !align_msb ) {
		// >>>>>>>A>>>>>>>B>>>>>>>C>>>>>>>D
		oval >>= 7;
	}
}

// 2x 4-bit --> 2x 8-bit (signed)
inline __device__ void gunpack(uint8_t  ival,
                               int16_t& oval,
                               bool     byte_reverse,
                               bool     align_msb,
                               bool     conjugate) {
	if( byte_reverse ) {
		// ........ABCDEFGH
		// EFGH....ABCD....
		oval = ival;
		oval = (oval | (oval << 12)) & 0xF0F0;
	} else {
		// ....ABCDEFGH....
		// ABCD....EFGH....
		oval = ival << 4;
		oval = (oval | (oval <<  4)) & 0xF0F0;
	}
	if( !align_msb ) {
		// >>>>ABCD>>>>EFGH
		rshift_subwords_gpu<4,int8_t>(oval);
	}
	if( conjugate ) {
		conjugate_subwords_gpu<int8_t>(oval);
	}
}
// 4x 2-bit --> 4x 8-bit (signed)
inline __device__ void gunpack(uint8_t  ival,
                               int32_t& oval,
                               bool     byte_reverse,
                               bool     align_msb,
                               bool     conjugate) {
	// ..................ABCDEFGH......
	// ......ABCD............EFGH......
	// AB......CD......EF......GH......
	oval = ival << 6;
	oval = (oval | (oval << 12)) & 0x03C003C0;
	oval = (oval | (oval <<  6)) & 0xC0C0C0C0;
	if( byte_reverse) {
		byteswap_gpu(oval, &oval);
	}
	if( !align_msb ) {
		// >>>>>>AB>>>>>>CD>>>>>>EF>>>>>>GH
		rshift_subwords_gpu<6,int8_t>(oval);
	}
	if( conjugate ) {
		conjugate_subwords_gpu<int8_t>(oval);
	}
}
// 8x 1-bit --> 8x 8-bit (signed)
inline __device__ void gunpack(uint8_t  ival,
                                        int64_t& oval,
                                        bool     byte_reverse,
                                        bool     align_msb,
                                        bool     conjugate) {
	// .................................................ABCDEFGH.......
	// .....................ABCD............................EFGH.......
	// .......AB..............CD..............EF..............GH.......
	// A.......B.......C.......D.......E.......F.......G.......H.......
	oval = (~ival) << 7;
	oval = (oval | (oval << 28)) & 0x0000078000000780;
	oval = (oval | (oval << 14)) & 0x0180018001800180;
	oval = (oval | (oval <<  7)) & 0x8080808080808080;
	oval |= 0x4040404040404040;
	if( byte_reverse) {
		byteswap_gpu(oval, &oval);
	}
	if( !align_msb ) {
		// >>>>>>>A>>>>>>>B>>>>>>>C>>>>>>>D
		rshift_subwords_gpu<6,int8_t>(oval);
	}
	if( conjugate ) {
		conjugate_subwords_gpu<int8_t>(oval);
	}
}

template<typename IType, typename OType>
struct GunpackFunctor {
	bool byte_reverse;
	bool align_msb;
	bool conjugate;
	GunpackFunctor(bool byte_reverse_,
	              bool align_msb_,
	              bool conjugate_)
		: byte_reverse(byte_reverse_),
		  align_msb(align_msb_),
		  conjugate(conjugate_) {}
	__device__ void operator()(IType ival, OType& oval) const {
		gunpack(ival, oval, byte_reverse, align_msb, conjugate);
	}
};

template<typename T, typename U, typename Func, typename Size>
__global__ void foreach_simple_gpu(T const* in,
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
                                      cudaStream_t stream) {
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

template<typename T, typename U, typename V, typename Func, typename Size>
__global__ void foreach_promote_gpu(T const* in,
                                    V*       out,
                                    Size     nelement,
                                    Func     func) {
	Size v0 = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x;
	
	if( v0 < nelement ) {
		U tmp2 = 0;
		func(in[v0], tmp2);
		for( Size j=0; j<sizeof(U)/sizeof(T); j++ ) {
			out[v0*sizeof(U)/sizeof(T) + j] = int8_t((tmp2 >> j*8) & 0xFF);
		}
	}
}

template<typename T, typename U, typename V, typename Func, typename Size>
inline void launch_foreach_promote_gpu(T const*     in,
                                       U*           tmp,
                                       V*           out,
                                       Size         nelement,
                                       Func         func,
                                       cudaStream_t stream) {
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
	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)foreach_promote_gpu<T,U,V,Func,Size>,
	                                         grid, block,
	                                         &args[0], 0, stream),
	                        BF_STATUS_INTERNAL_ERROR);
}

// Instantiation - Gunpack functors used in unpack.cpp
//// unsigned
template class GunpackFunctor<uint8_t,uint16_t>;
template class GunpackFunctor<uint8_t,uint32_t>;
template class GunpackFunctor<uint8_t,uint64_t>;
//// signed
template class GunpackFunctor<uint8_t,int16_t>;
template class GunpackFunctor<uint8_t,int32_t>;
template class GunpackFunctor<uint8_t,int64_t>;

// Instantiation - launch_foreach_simple_gpu calls used in unpack.cpp
//// unsigned
template void launch_foreach_simple_gpu<uint8_t,uint16_t,GunpackFunctor<uint8_t,uint16_t>,size_t>(uint8_t const* in,
                                                                                                  uint16_t*      out,
                                                                                                  size_t         nelement,
                                                                                                  GunpackFunctor<uint8_t,uint16_t> func,
                                                                                                  cudaStream_t   stream);
template void launch_foreach_simple_gpu<uint8_t,uint32_t,GunpackFunctor<uint8_t,uint32_t>,size_t>(uint8_t const* in,
                                                                                                  uint32_t*      out,
                                                                                                  size_t         nelement,
                                                                                                  GunpackFunctor<uint8_t,uint32_t> func,
                                                                                                  cudaStream_t   stream);
template void launch_foreach_simple_gpu<uint8_t,uint64_t,GunpackFunctor<uint8_t,uint64_t>,size_t>(uint8_t const* in,
                                                                                                  uint64_t*      out,
                                                                                                  size_t         nelement,
                                                                                                  GunpackFunctor<uint8_t,uint64_t> func,
                                                                                                  cudaStream_t   stream);
//// signed
template void launch_foreach_simple_gpu<uint8_t,int16_t,GunpackFunctor<uint8_t,int16_t>,size_t>(uint8_t const* in,
                                                                                                int16_t*       out,
                                                                                                size_t         nelement,
                                                                                                GunpackFunctor<uint8_t,int16_t> func,
                                                                                                cudaStream_t   stream);
template void launch_foreach_simple_gpu<uint8_t,int32_t,GunpackFunctor<uint8_t,int32_t>,size_t>(uint8_t const* in,
                                                                                                int32_t*       out,
                                                                                                size_t         nelement,
                                                                                                GunpackFunctor<uint8_t,int32_t> func,
                                                                                                cudaStream_t   stream);
template void launch_foreach_simple_gpu<uint8_t,int64_t,GunpackFunctor<uint8_t,int64_t>,size_t>(uint8_t const *in,
                                                                                                int64_t*       out,
                                                                                                size_t         nelement,
                                                                                                GunpackFunctor<uint8_t,int64_t> func,
                                                                                                cudaStream_t   stream);

// Instantiation - launch_foreach_promote_gpu calls used in unpack.cpp
//// promote to float
template void launch_foreach_promote_gpu<uint8_t,int16_t,float,GunpackFunctor<uint8_t,int16_t>,size_t>(uint8_t const* in,
                                                                                                       int16_t*       tmp,
                                                                                                       float*         out,
                                                                                                       size_t         nelement,
                                                                                                       GunpackFunctor<uint8_t,int16_t> func,
                                                                                                       cudaStream_t   stream);
template void launch_foreach_promote_gpu<uint8_t,int32_t,float,GunpackFunctor<uint8_t,int32_t>,size_t>(uint8_t const* in,
                                                                                                       int32_t*       tmp,
                                                                                                       float*         out,
                                                                                                       size_t         nelement,
                                                                                                       GunpackFunctor<uint8_t,int32_t> func,
                                                                                                       cudaStream_t   stream);
template void launch_foreach_promote_gpu<uint8_t,int64_t,float,GunpackFunctor<uint8_t,int64_t>,size_t>(uint8_t const* in,
                                                                                                       int64_t*       tmp,
                                                                                                       float*         out,
                                                                                                       size_t         nelement,
                                                                                                       GunpackFunctor<uint8_t,int64_t> func,
                                                                                                       cudaStream_t   stream);
//// promote to double
template void launch_foreach_promote_gpu<uint8_t,int16_t,double,GunpackFunctor<uint8_t,int16_t>,size_t>(uint8_t const* in,
                                                                                                        int16_t*       tmp,
                                                                                                        double*        out,
                                                                                                        size_t         nelement,
                                                                                                        GunpackFunctor<uint8_t,int16_t> func,
                                                                                                        cudaStream_t   stream);
template void launch_foreach_promote_gpu<uint8_t,int32_t,double,GunpackFunctor<uint8_t,int32_t>,size_t>(uint8_t const* in,
                                                                                                        int32_t*       tmp,
                                                                                                        double*        out,
                                                                                                        size_t         nelement,
                                                                                                        GunpackFunctor<uint8_t,int32_t> func,
                                                                                                        cudaStream_t   stream);
template void launch_foreach_promote_gpu<uint8_t,int64_t,double,GunpackFunctor<uint8_t,int64_t>,size_t>(uint8_t const* in,
                                                                                                        int64_t*       tmp,
                                                                                                        double*        out,
                                                                                                        size_t         nelement,
                                                                                                        GunpackFunctor<uint8_t,int64_t> func,
                                                                                                        cudaStream_t   stream);

