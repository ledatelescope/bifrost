/*
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
