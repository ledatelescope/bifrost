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

#include <bifrost/transpose.h>
#include "assert.hpp"

#if BF_CUDA_ENABLED
  #include "transpose_gpu_kernel.cuh"
  #include "cuda/stream.hpp"
#else
  typedef int cudaStream_t; // WAR
#endif

#include <cstdio>
#include <algorithm>
#include <limits>

#include <cassert> // TODO: TESTING ONLY

template<int N> struct aligned_type     { typedef char  type; };
template<>      struct aligned_type< 2> { typedef short type; };
template<>      struct aligned_type< 4> { typedef int   type; };
template<>      struct aligned_type< 8> { typedef int2  type; };
template<>      struct aligned_type<16> { typedef int4  type; };

namespace typed {
template<typename T>
inline T gcd(T u, T v) {
	return (v == 0) ? u : gcd(v, u % v);
}
namespace aligned_in {
namespace aligned_in_out {

template<typename T>
T div_up(T n, T d) {
	return (n-1)/d+1;
}
template<typename T>
bool is_pow2(T v) {
	return v && !(v & (v - 1));
}
template<typename T>
T log2(T v) {
	T r;
	T shift;
	r =     (v > 0xFFFFFFFF) << 5; v >>= r;
	shift = (v > 0xFFFF    ) << 4; v >>= shift; r |= shift;
	shift = (v > 0xFF      ) << 3; v >>= shift; r |= shift;
	shift = (v > 0xF       ) << 2; v >>= shift; r |= shift;
	shift = (v > 0x3       ) << 1; v >>= shift; r |= shift;
	                                            r |= (v >> 1);
	return r;
}

//inline int find_odim(BFsize const* output_order, int ndim, int dim) {
//	return std::find(output_order, output_order+ndim, dim) - output_order;
//}
#define FIND_ODIM_(dim)	\
	(int(std::find(output_order, output_order+ndim, (dim)) - output_order))

template<int ALIGNMENT_IN, int ALIGNMENT_OUT,
         typename T>
int transpose(BFsize        ndim,
              BFsize const* sizes,          // elements
              BFsize const* output_order,
              T      const* in,
              BFsize const* in_strides,  // bytes
              T           * out,
              BFsize const* out_strides, // bytes
              cudaStream_t  stream) {
	enum { ELEMENT_SIZE = sizeof(T) };
	// TODO: This is currently all tuned for a GTX Titan (sm_35)
	enum {
		TILE_DIM   = 32,
		//BLOCK_ROWS = 8
		//BLOCK_ROWS = (ALIGNMENT_IN >= 8 && ALIGNMENT_IN % 2 == 0) ? 16 : 8
		BLOCK_ROWS = (ELEMENT_SIZE >= 8 && ELEMENT_SIZE % 2 == 0) ? 16 : 8,
		CONDITIONAL_WRITE = (ELEMENT_SIZE != 1 &&
		                     ELEMENT_SIZE != 2)
	};
	
	//// Note: These are all relative to the input
	int ifastdim = ndim-1;
	int islowdim = output_order[ndim-1];
	int oslowdim = FIND_ODIM_(ifastdim);
	//int ofastdim = islowdim;
	//int oslowdim = ndim-1;//find_odim(output_order, ndim, ndim-1);
		//std::find(output_order, output_order+ndim, ndim-1)-output_order;
	//std::printf("****ifast,islow,ofast,oslow: %i %i %i %i\n",
	//            ifastdim, islowdim,
	//            ofastdim, oslowdim);
	if( ifastdim == islowdim ) {//ofastdim ) {
		// TODO: Use plain permute-copy kernel
		std::printf("PLAIN PERMUTATIONS NOT IMPLEMENTED YET (%i %i)\n",
		            islowdim, ifastdim);//oslowdim);
		//return -1;
		return BF_STATUS_UNSUPPORTED;
	}
	BFsize width       = sizes[ifastdim];
	BFsize height      = sizes[islowdim];//ofastdim];
	BFsize  in_stride  =  in_strides[islowdim];
	BFsize out_stride  = out_strides[oslowdim];//oslowdim)]; // TODO
	
	BFsize ndimz = std::max(ndim-2, BFsize(0));
	int    shapez[MAX_NDIM];
	int    istridez[MAX_NDIM];
	int    ostridez[MAX_NDIM];
	BFsize sizez = 1;
	int dd = 0;
	for( int d=0; d<ndim-1; ++d ) {
		if( d != islowdim ) {
			shapez[dd]   = sizes[d];
			sizez       *= sizes[d];
			istridez[dd] = in_strides[d];
			ostridez[dd] = out_strides[FIND_ODIM_(d)]; // TODO: Check this
			//std::printf("shapez, sizez, istridez, ostridez = %i %lu %i %i\n",
			//            shapez[dd], sizez, istridez[dd], ostridez[dd]);
			++dd;
		}
	}
	int cumushapez[MAX_NDIM];
	cumushapez[ndimz-1] = 1;
	for( int d=ndimz-2; d>=0; --d ) {
		cumushapez[d] = cumushapez[d+1]*shapez[d+1];
	}
	
	dim3 grid, block;
	block.x = TILE_DIM;
	block.y = BLOCK_ROWS;
	block.z = 1;
	grid.x = std::min(div_up(width,  (BFsize)TILE_DIM), (BFsize)65535);
	grid.y = std::min(div_up(height, (BFsize)TILE_DIM), (BFsize)65535);
	// Note: inner_idepth*outer_idepth == inner_odepth*outer_odepth
	//grid.z = std::min(inner_idepth*outer_idepth,        (BFsize)65535);
	grid.z = std::min(sizez,        (BFsize)65535);
	//std::printf("ALIGN IN:  %i %lu\n", ALIGNMENT_IN, sizeof(typename aligned_type<ALIGNMENT_IN>::type));
	//std::printf("ALIGN ouT: %i %lu\n", ALIGNMENT_OUT, sizeof(typename aligned_type<ALIGNMENT_OUT>::type));
	
	bool can_use_int = (sizes[0]*in_strides[0] <
	                    (BFsize)std::numeric_limits<int>::max() &&
	                    sizes[0]*out_strides[0] <
	                    (BFsize)std::numeric_limits<int>::max());
#if BF_CUDA_ENABLED
	if( ELEMENT_SIZE ==  6 ||
	    ELEMENT_SIZE ==  8 ||
	    ELEMENT_SIZE == 16 ) {
		// TODO: Doing this here might be a bad idea
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	}
	else {
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	}
	if( can_use_int ) {
		kernel::transpose
			<TILE_DIM,BLOCK_ROWS,CONDITIONAL_WRITE,
			 typename aligned_type<ALIGNMENT_IN>::type,
			 typename aligned_type<ALIGNMENT_OUT>::type,
			 T, int>
			<<<grid,block,0,stream>>>(width, height,
			                          in, in_stride,
			                          out, out_stride,
			                          sizez, ndimz,
			                          cumushapez,
			                          istridez,
			                          ostridez);
			                          /*
			                          inner_idepth,
			                          inner_idepth_log2,
			                          inner_odepth,
			                          inner_odepth_log2,
			                          outer_idepth,
			                          outer_odepth,
			                          inner_ipitch,
			                          inner_opitch,
			                          outer_ipitch,
			                          outer_opitch);
			                          */
		//inner_depth_istride,
		//inner_depth_ostride,
		//outer_depth_istride,
		//outer_depth_ostride);
	}
	else {
		kernel::transpose
			<TILE_DIM,BLOCK_ROWS,CONDITIONAL_WRITE,
			 typename aligned_type<ALIGNMENT_IN>::type,
			 typename aligned_type<ALIGNMENT_OUT>::type,
			 T, BFsize>
			<<<grid,block,0,stream>>>(width, height,
			                          in, in_stride,
			                          out, out_stride,
			                          sizez, ndimz,
			                          cumushapez,
			                          istridez,
			                          ostridez);
			                          /*
			                          inner_idepth,
			                          inner_idepth_log2,
			                          inner_odepth,
			                          inner_odepth_log2,
			                          outer_idepth,
			                          outer_odepth,
			                          inner_ipitch,
			                          inner_opitch,
			                          outer_ipitch,
			                          outer_opitch);
			                          */
			                          //inner_depth,
			                          //inner_depth_log2,
			                          //outer_depth,
			                          //inner_depth_istride,
			                          //inner_depth_ostride,
			                          //outer_depth_istride,
			                          //outer_depth_ostride);
	}
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess ) {
		std::printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
	}
	BF_ASSERT(error == cudaSuccess, BF_STATUS_INTERNAL_ERROR);
#endif
	/*
	// TODO: Implement CPU version too
#pragma omp parallel for collapse(3)
	for( int blockIdx_z=0; blockIdx_z<gridDim.z; ++blockIdx_z ) {
	for( int blockIdx_y=0; blockIdx_y<gridDim.y; ++blockIdx_y ) {
	for( int blockIdx_x=0; blockIdx_x<gridDim.x; ++blockIdx_x ) {
		for( int threadIdx_z=0; threadIdx_z<blockDim.z; ++threadIdx_z ) {
		for( int threadIdx_y=0; threadIdx_y<blockDim.y; ++threadIdx_y ) {
		for( int threadIdx_x=0; threadIdx_x<blockDim.x; ++threadIdx_x ) {
			transpose<...>(...);
		}
		}
		}
	}
	}
	}
	*/
	return BF_STATUS_SUCCESS;
}
} // namespace aligned_in_out
template<int ALIGNMENT_IN, typename T>
int transpose(BFsize        ndim,
              BFsize const* sizes,       // elements
              BFsize const* output_order,
              T      const* in,
              BFsize const* in_strides,  // bytes
              T           * out,
              BFsize const* out_strides, // bytes
              cudaStream_t  stream) {
	BFsize out_alignment = (BFsize)out;
	for( int d=0; d<ndim; ++d ) {
		out_alignment = gcd(out_alignment, out_strides[d]);
	}
	switch( out_alignment ) {
	case 16: return aligned_in_out::transpose<ALIGNMENT_IN,16>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	case  8: return aligned_in_out::transpose<ALIGNMENT_IN, 8>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	case 12:
	case  4: return aligned_in_out::transpose<ALIGNMENT_IN, 4>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	case 14:
	case 10:
	case  6:
	case  2: return aligned_in_out::transpose<ALIGNMENT_IN, 2>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	default: return aligned_in_out::transpose<ALIGNMENT_IN, 1>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	}
}
} // namespace aligned_in
template<typename T>
int transpose(BFsize        ndim,
              BFsize const* sizes,       // elements
              BFsize const* output_order,
              T      const* in,
              BFsize const* in_strides,  // bytes
              T           * out,
              BFsize const* out_strides, // bytes
              cudaStream_t  stream) {
	BFsize in_alignment = (BFsize)in;
	for( int d=0; d<ndim; ++d ) {
		in_alignment = gcd(in_alignment, in_strides[d]);
	}
	switch( in_alignment ) {
	case 16: return aligned_in::transpose<16>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	case  8: return aligned_in::transpose< 8>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	case 12:
	case  4: return aligned_in::transpose< 4>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	case 14:
	case 10:
	case  6:
	case  2: return aligned_in::transpose< 2>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	default: return aligned_in::transpose< 1>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	}
}
} // namespace typed

BFstatus bfTranspose(void*         dst,
                     BFsize const* dst_strides,
                     void   const* src,
                     BFsize const* src_strides,
                     BFspace       space,
                     BFsize        element_size,
                     BFsize        ndim,
                     BFsize const* src_shape,
                     BFsize const* axes) {
	BF_ASSERT(dst && src, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(ndim >= 2, BF_STATUS_INVALID_ARGUMENT);
	// TODO: Implement BF_SPACE_AUTO
	BF_ASSERT(space == BF_SPACE_CUDA ||
	          space == BF_SPACE_CUDA_MANAGED,
	          BF_STATUS_UNSUPPORTED);
	//for( int d=0; d<ndim; ++d ) {
	//	std::cout << dst_strides[d] << "\t"
	//	          << src_strides[d] << "\t"
	//	          << src_shape[d] << "\t"
	//	          << axes[d] << std::endl;
	//}
	cuda::scoped_stream stream;
	BFsize istrides_actual[MAX_NDIM];
	BFsize ostrides_actual[MAX_NDIM];
	if( src_strides ) {
		// TODO: Consider supporting strides on fastest dims
		//assert( in_strides[ndim-1] == element_size );
		BF_ASSERT(src_strides[ndim-1] == element_size,
		          BF_STATUS_UNSUPPORTED);
		for( int d=0; d<(int)ndim; ++d ) {
			istrides_actual[d] = src_strides[d];
		}
	}
	else {
		istrides_actual[ndim-1] = element_size;
		for( int d=(int)ndim-2; d>=0; --d ) {
			istrides_actual[d] = istrides_actual[d+1] * src_shape[d+1];
		}
	}
	if( dst_strides ) {
		//assert( dst_strides[ndim-1] == element_size );
		BF_ASSERT(dst_strides[ndim-1] == element_size,
		          BF_STATUS_UNSUPPORTED);
		for( int d=0; d<(int)ndim; ++d ) {
			ostrides_actual[d] = dst_strides[d];
		}
	}
	else {
		ostrides_actual[ndim-1] = element_size;
		for( int d=(int)ndim-2; d>=0; --d ) {
			ostrides_actual[d] = ostrides_actual[d+1] * src_shape[axes[d+1]];
		}
	}
	switch( element_size ) {
#define DEFINE_TYPE_CASE(N)	  \
	case N: return typed::transpose(ndim,src_shape,axes,(type_of_size<N>*)src,istrides_actual,(type_of_size<N>*)dst,ostrides_actual,stream);
		DEFINE_TYPE_CASE( 1); DEFINE_TYPE_CASE( 2); DEFINE_TYPE_CASE( 3); DEFINE_TYPE_CASE( 4);
		DEFINE_TYPE_CASE( 5); DEFINE_TYPE_CASE( 6); DEFINE_TYPE_CASE( 7); DEFINE_TYPE_CASE( 8);
		DEFINE_TYPE_CASE( 9); DEFINE_TYPE_CASE(10); DEFINE_TYPE_CASE(11); DEFINE_TYPE_CASE(12);
		DEFINE_TYPE_CASE(13); DEFINE_TYPE_CASE(14); DEFINE_TYPE_CASE(15); DEFINE_TYPE_CASE(16);
#undef DEFINE_TYPE_CASE
	case 0: return BF_STATUS_SUCCESS; // Do nothing on zero-size data
	//default: std::printf("UNSUPPORTED ELEMENT SIZE\n"); return -1;
	default: BF_ASSERT(false, BF_STATUS_UNSUPPORTED);
	}
}
