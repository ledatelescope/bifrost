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

#pragma once

#include "int_fastdiv.h"

enum { MAX_NDIM = 8 };

template<typename T, int N>
class SmallArray {
	T vals_[N];
public:
	template<typename Y>
	SmallArray(Y vals[N]) {
		for( int i=0; i<N; ++i ) {
			vals_[i] = vals[i];
		}
	}
	inline __host__ __device__
	T operator[](int i) const {
		return vals_[i];
	}
};

//namespace typed {
//namespace aligned_in {
//namespace aligned_in_out {

#pragma pack(1)
template<int N> struct /*__align__(1)*/ type_of_size { char _[N]; };

namespace kernel {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
#define LDG(value) __ldg(&(value))
#else
#define LDG(value) (value)
#endif
template<int TILE_DIM, int BLOCK_ROWS,
         bool CONDITIONAL_WRITE,
         typename T_LD,
         typename T_ST,
         typename T,
         typename SizeT=size_t>
__global__
void transpose(int      width,        // elements
               int      height,       // elements
               const T* __restrict__ in,
               SizeT   in_stride,    // bytes
               T*       __restrict__ out,
               SizeT   out_stride,
               int     sizez,
               int     ndimz,
               SmallArray<int_fastdiv,MAX_NDIM> cumushapez,
               SmallArray<int,MAX_NDIM>         istridez,
               SmallArray<int,MAX_NDIM>         ostridez) {
	
	__shared__ T tile[TILE_DIM][TILE_DIM+1]; // +1 to avoid bank conflicts
	
	for( int blockIdx_z=blockIdx.z; blockIdx_z<sizez; blockIdx_z+=gridDim.z ) {
	int z = blockIdx_z;
	SizeT zoffset_in  = 0;
	SizeT zoffset_out = 0;
	for( int d=0; d<ndimz; ++d ) {
		int idx = z / cumushapez[d];
		z -=      idx*cumushapez[d];
		zoffset_in  += idx*istridez[d];
		zoffset_out += idx*ostridez[d];
	}
	for( int blockIdx_y=blockIdx.y; blockIdx_y*TILE_DIM<height; blockIdx_y+=gridDim.y ) {
		for( int blockIdx_x=blockIdx.x; blockIdx_x*TILE_DIM<width; blockIdx_x+=gridDim.x ) {
			int ix = blockIdx_x*TILE_DIM + threadIdx.x;
			int iy = blockIdx_y*TILE_DIM + threadIdx.y;
			//int excess_ix = abs(ix - (width-1));
			ix = min(ix, width-1);
			__syncthreads();
#pragma unroll
			for( int r=0; r<TILE_DIM; r+=BLOCK_ROWS ) {
				int iyr = iy+r;
				//if( ix >= width || iyr >= height ) continue; // Avoid excess threads
				iyr = min(iyr, height-1);
				T val;
				//int byte_offset = ix*sizeof(T) + iyr*in_stride;
				SizeT byte_offset = ix*sizeof(T) + iyr*in_stride + zoffset_in;
				// Note: This deals with misaligned addresses when necessary
#pragma unroll
				for( int k=0; k<sizeof(T)/sizeof(T_LD); ++k ) {
					((T_LD*)&val)[k] = LDG( ((T_LD*)in)[byte_offset/sizeof(T_LD)+k] );
					/*
					  TODO: If sizeof(T_LD) > 8, load 8-bytes per thread x2 and write directly to smem
					  TODO: Could have first and last threads load <4 bytes and then the rest load aligned4?
					 */
				}
				tile[threadIdx.y+r][threadIdx.x] = val;
			}
			__syncthreads();
			int ox = blockIdx_y*TILE_DIM + threadIdx.x;
			if( ox >= height ) continue; // Cull excess threads
			//ox = min(ox, height-1);
			int oy = blockIdx_x*TILE_DIM + threadIdx.y;
#pragma unroll
			for( int c=0; c<TILE_DIM; c+=BLOCK_ROWS ) {
				int oyc = oy+c;
				if( CONDITIONAL_WRITE ) {
					if( oyc >= width ) continue;
				}
				else {
					oyc = min(oyc, width-1); // Better for sizeof(T) = 1,2
				}
				T val = tile[threadIdx.x][threadIdx.y+c];
				//int byte_offset = ox*sizeof(T) + oyc*out_stride;
				SizeT byte_offset = ox*sizeof(T) + oyc*out_stride + zoffset_out;
#pragma unroll
				for( int k=0; k<sizeof(T)/sizeof(T_ST); ++k ) {
					((T_ST*)out)[byte_offset/sizeof(T_ST)+k] = ((T_ST*)&val)[k];
				}
			}
		}
	}
	} // blockIdx_z
}
#undef LDG
} // namespace kernel

//} // namespace aligned_out
//} // namespace aligned_in
//} // namespace typed
