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

#include <bifrost/transpose.h>
#include <bifrost/map.h>
#include "assert.hpp"
#include "utils.hpp"
#include "trace.hpp"

#if BF_CUDA_ENABLED
  #include "transpose_gpu_kernel.cuh"
  #include "cuda.hpp"
#else
  typedef int cudaStream_t; // WAR
#endif

#include <cstdio>
#include <algorithm>
#include <limits>
#include <sstream>

template<int N> struct aligned_type     { typedef char  type; };
template<>      struct aligned_type< 2> { typedef short type; };
template<>      struct aligned_type< 4> { typedef int   type; };
template<>      struct aligned_type< 8> { typedef int2  type; };
template<>      struct aligned_type<16> { typedef int4  type; };

//inline int find_odim(int const* output_order, int ndim, int dim) {
//	return std::find(output_order, output_order+ndim, dim) - output_order;
//}
#define FIND_ODIM_(dim)	\
	(int(std::find(output_order, output_order+ndim, (dim)) - output_order))

namespace typed {
namespace aligned_in {
namespace aligned_in_out {

template<int ALIGNMENT_IN, int ALIGNMENT_OUT,
         typename T>
BFstatus transpose(int           ndim,
                   long   const* sizes,          // elements
                   int    const* output_order,
                   T      const* in,
                   long   const* in_strides,  // bytes
                   T           * out,
                   long   const* out_strides, // bytes
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
		return BF_STATUS_UNSUPPORTED;
	}
	long width       = sizes[ifastdim];
	long height      = sizes[islowdim];//ofastdim];
	long  in_stride  =  in_strides[islowdim];
	long out_stride  = out_strides[oslowdim];//oslowdim)]; // TODO
	
	int  ndimz = std::max(ndim-2, int(0));
	int  shapez[MAX_NDIM];
	int  istridez[MAX_NDIM];
	int  ostridez[MAX_NDIM];
	long sizez = 1;
	int dd = 0;
	for( int d=0; d<(int)ndim-1; ++d ) {
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
	for( int d=MAX_NDIM-1; d>ndimz-2; --d ) {
		// WAR to avoid uninitialized values going through int_fastdiv
		//   via SmallArray, which triggers errors when run through valgrind.
		cumushapez[d] = 1;
	}
	
	dim3 grid, block;
	block.x = TILE_DIM;
	block.y = BLOCK_ROWS;
	block.z = 1;
	grid.x = std::min(div_up(width,  (long)TILE_DIM), (long)65535);
	grid.y = std::min(div_up(height, (long)TILE_DIM), (long)65535);
	// Note: inner_idepth*outer_idepth == inner_odepth*outer_odepth
	//grid.z = std::min(inner_idepth*outer_idepth,        (long)65535);
	grid.z = std::min(sizez,        (long)65535);
	//std::printf("ALIGN IN:  %i %lu\n", ALIGNMENT_IN, sizeof(typename aligned_type<ALIGNMENT_IN>::type));
	//std::printf("ALIGN ouT: %i %lu\n", ALIGNMENT_OUT, sizeof(typename aligned_type<ALIGNMENT_OUT>::type));
	
	bool can_use_int = (sizes[0]*in_strides[0] <
	                    (long)std::numeric_limits<int>::max() &&
	                    sizes[0]*out_strides[0] <
	                    (long)std::numeric_limits<int>::max());
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
			 T, long>
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
BFstatus transpose(int           ndim,
                   long   const* sizes,       // elements
                   int    const* output_order,
                   T      const* in,
                   long   const* in_strides,  // bytes
                   T           * out,
                   long   const* out_strides, // bytes
                   cudaStream_t  stream) {
	unsigned long out_alignment = (unsigned long)out;
	for( int d=0; d<ndim; ++d ) {
		out_alignment = gcd(out_alignment, (unsigned long)out_strides[d]);
	}
	switch( out_alignment ) {
	case sizeof(T): return aligned_in_out::transpose<ALIGNMENT_IN,sizeof(T)>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	//case 16: return aligned_in_out::transpose<ALIGNMENT_IN,16>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	//case  8: return aligned_in_out::transpose<ALIGNMENT_IN, 8>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	//case 12:
	//case  4: return aligned_in_out::transpose<ALIGNMENT_IN, 4>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	//case 14:
	//ncase 10:
	//case  6:
	//case  2: return aligned_in_out::transpose<ALIGNMENT_IN, 2>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	default: return aligned_in_out::transpose<ALIGNMENT_IN, 1>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	}
}
} // namespace aligned_in
template<typename T>
BFstatus transpose(int            ndim,
                   long    const* sizes,       // elements
                   int     const* output_order,
                   T       const* in,
                   long    const* in_strides,  // bytes
                   T            * out,
                   long    const* out_strides, // bytes
                   cudaStream_t   stream) {
	BF_TRACE_STREAM(stream);
	unsigned long in_alignment = (unsigned long)in;
	for( int d=0; d<ndim; ++d ) {
		in_alignment = gcd(in_alignment, (unsigned long)in_strides[d]);
	}
	switch( in_alignment ) {
	case sizeof(T): return aligned_in::transpose<sizeof(T)>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	//case 16: return aligned_in::transpose<16>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	//case  8: return aligned_in::transpose< 8>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	//case 12:
	//case  4: return aligned_in::transpose< 4>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	//case 14:
	//case 10:
	//case  6:
	//case  2: return aligned_in::transpose< 2>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	default: return aligned_in::transpose< 1>(ndim,sizes,output_order,in,in_strides,out,out_strides,stream);
	}
}
} // namespace typed

// This is for when the fastest-changing dim is not transposed
BFstatus transpose_simple(BFarray const* in,
                          BFarray const* out,
                          int     const* axes) {
	BF_ASSERT(BF_MAX_DIMS <= 16, BF_STATUS_INTERNAL_ERROR);
	// Minor HACK to avoid using stringstream (which was inexplicably
	//   segfaulting whenever I used it :|).
	static const char* hex_digits = "0123456789ABCDEF";
	int ndim = in->ndim;
	
	int axes_inverted[BF_MAX_DIMS];
	invert_permutation(ndim, axes, axes_inverted);
	axes = axes_inverted;
	
	std::string func_str;
	func_str += "out = in(i";
	func_str += hex_digits[axes[0]];
	for( int d=1; d<ndim; ++d ) {
		func_str += ", i";
		func_str += hex_digits[axes[d]];
	}
	func_str += ")";
	// Minor HACK to avoid heap allocations
	char const* axis_names[] = {
		"i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7",
		"i8", "i9", "iA", "iB", "iC", "iD", "iE", "iF"
	};
	BFarray in_mutable, out_mutable;
	::memcpy( &in_mutable,  in, sizeof(BFarray));
	::memcpy(&out_mutable, out, sizeof(BFarray));
	in_mutable.immutable = true;
	in_mutable.dtype  = same_sized_storage_dtype(in_mutable.dtype);
	out_mutable.dtype = same_sized_storage_dtype(out_mutable.dtype);
	in = &in_mutable;
	out = &out_mutable;
	int narg = 2;
	BFarray const* args[] = {in, out};
	char const* arg_names[] = {"in", "out"};
	char const* func = func_str.c_str();
	char const* extra_code = 0;
	return bfMap(ndim, out->shape, axis_names, narg, args, arg_names,
	             "transpose_simple", func, extra_code, 0, 0);
}

// This is for when the fastest-changing input dim is small and should
//   be processed whole by each thread using vector loads.
BFstatus transpose_vector_read(BFarray const* in,
                               BFarray const* out,
                               int     const* axes) {
	BF_ASSERT(BF_MAX_DIMS <= 16, BF_STATUS_INTERNAL_ERROR);
	// Minor HACK to avoid using stringstream (which was inexplicably
	//   segfaulting whenever I used it :|).
	static const char* hex_digits = "0123456789ABCDEF";
	int ndim = in->ndim;
	BF_ASSERT(in->shape[ndim-1] <= 16, BF_STATUS_INTERNAL_ERROR);
	int K = in->shape[ndim-1];
	
	int axes_inverted[BF_MAX_DIMS];
	invert_permutation(ndim, axes, axes_inverted);
	axes = axes_inverted;
	
	int odim = axes[ndim-1];
	std::string in_inds_str = "i";
	in_inds_str += hex_digits[axes[0]];
	std::string out_inds_str = (odim == 0) ? "k" : "i0";
	for( int d=1; d<ndim; ++d ) {
		if( d < ndim-1 ) {
			in_inds_str += ", i";
			in_inds_str += hex_digits[axes[d]];
		}
		out_inds_str += ", ";
		if( d == odim ) {
			out_inds_str += "k";
		} else {
			out_inds_str += "i";
			out_inds_str += hex_digits[d];
		}
	}
	std::string func_str;
	func_str += "enum { K = " + std::to_string(K) + " };\n";
	func_str +=
		"in_type ivals = in(" + in_inds_str + ");\n"
		"#pragma unroll\n"
		"for( int k=0; k<K; ++k ) {\n"
		"    out(" + out_inds_str + ") = ivals[k];\n"
		"}\n";
	// Minor HACK to avoid heap allocations
	char const* axis_names[] = {
		"i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7",
		"i8", "i9", "iA", "iB", "iC", "iD", "iE", "iF"
	};
	BFarray in_mutable, out_mutable;
	::memcpy( &in_mutable,  in, sizeof(BFarray));
	::memcpy(&out_mutable, out, sizeof(BFarray));
	in_mutable.immutable = true;
	in_mutable.dtype  = same_sized_storage_dtype(in_mutable.dtype);
	out_mutable.dtype = same_sized_storage_dtype(out_mutable.dtype);
	merge_last_dim_into_dtype(&in_mutable, &in_mutable);
	in  = &in_mutable;
	out = &out_mutable;
	int narg = 2;
	BFarray const* args[] = {in, out};
	char const* arg_names[] = {"in", "out"};
	char const* func = func_str.c_str();
	char const* extra_code = 0;
	long shape[BF_MAX_DIMS];
	::memcpy(shape, out->shape, ndim*sizeof(long));
	shape[odim] = 1; // This dim is processed sequentially by each thread
	return bfMap(ndim, shape, axis_names, narg, args, arg_names,
	             "transpose_vector_read", func, extra_code, 0, 0);
}

// This is for when the fastest-changing output dim is small and should
//   be processed whole by each thread using vector stores.
BFstatus transpose_vector_write(BFarray const* in,
                                BFarray const* out,
                                int     const* axes) {
	BF_ASSERT(BF_MAX_DIMS <= 16, BF_STATUS_INTERNAL_ERROR);
	// Minor HACK to avoid using stringstream (which was inexplicably
	//   segfaulting whenever I used it :|).
	static const char* hex_digits = "0123456789ABCDEF";
	int ndim = in->ndim;
	BF_ASSERT(out->shape[ndim-1] <= 16, BF_STATUS_INTERNAL_ERROR);
	int K = out->shape[ndim-1];
	
	int axes_inverted[BF_MAX_DIMS];
	invert_permutation(ndim, axes, axes_inverted);
	axes = axes_inverted;
	
	int idim = ndim - 1;
	std::string in_inds_str;
	if( idim == axes[0] ) {
		in_inds_str = "k";
	} else {
		in_inds_str = "i";
		in_inds_str += hex_digits[axes[0]];
	}
	std::string out_inds_str = "i0";
	for( int d=1; d<ndim; ++d ) {
		in_inds_str += ", ";
		if( axes[d] == idim ) {
			in_inds_str += "k";
		} else {
			in_inds_str += "i";
			in_inds_str += hex_digits[axes[d]];
		}
		if( d < ndim-1 ) {
			out_inds_str += ", i";
			out_inds_str += hex_digits[d];
		}
	}
	std::string func_str;
	func_str += "enum { K = " + std::to_string(K) + " };\n";
	func_str +=
		"out_type ovals;\n"
		"#pragma unroll\n"
		"for( int k=0; k<K; ++k ) {\n"
		"    ovals[k] = in(" + in_inds_str + ");\n"
		"}\n"
		"out(" + out_inds_str + ") = ovals;\n";
	// Minor HACK to avoid heap allocations
	char const* axis_names[] = {
		"i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7",
		"i8", "i9", "iA", "iB", "iC", "iD", "iE", "iF"
	};
	long shape[BF_MAX_DIMS];
	::memcpy(shape, out->shape, ndim*sizeof(long));
	shape[idim] = 1; // This dim is processed sequentially by each thread
	BFarray in_mutable, out_mutable;
	::memcpy( &in_mutable,  in, sizeof(BFarray));
	::memcpy(&out_mutable, out, sizeof(BFarray));
	in_mutable.immutable = true;
	in_mutable.dtype  = same_sized_storage_dtype(in_mutable.dtype);
	out_mutable.dtype = same_sized_storage_dtype(out_mutable.dtype);
	merge_last_dim_into_dtype(&out_mutable, &out_mutable);
	in  = &in_mutable;
	out = &out_mutable;
	int narg = 2;
	BFarray const* args[] = {in, out};
	char const* arg_names[] = {"in", "out"};
	char const* func = func_str.c_str();
	char const* extra_code = 0;
	return bfMap(ndim, shape, axis_names, narg, args, arg_names,
	             "transpose_vector_write", func, extra_code, 0, 0);
}

BFstatus bfTranspose(BFarray const* in,
                     BFarray const* out,
                     int     const* axes) {
	BF_TRACE();
	BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
	BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
	BF_ASSERT(axes, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(in->ndim >= 2,         BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(out->ndim == in->ndim, BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(space_accessible_from(in->space, BF_SPACE_CUDA),
	          BF_STATUS_UNSUPPORTED_SPACE);
	BF_ASSERT(in->dtype == out->dtype, BF_STATUS_INVALID_DTYPE);
	
	int element_size = BF_DTYPE_NBYTE(in->dtype);
	int ndim = in->ndim;
	
	// Handle negative axis numbers
	int axes_actual[BF_MAX_DIMS];
	for( int d=0; d<ndim; ++d ) {
		int x = axes[d];
		axes_actual[d] = x < 0 ? ndim + x : x;
		BF_ASSERT(out->shape[d] == in->shape[axes_actual[d]],
		          BF_STATUS_INVALID_SHAPE);
	}
	
	// Special cases to be handled with different kernels
	int ifastdim = ndim-1;
	int ofastdim = axes_actual[ndim-1];
	if( ifastdim == ofastdim ) {
		return transpose_simple(in, out, axes_actual);
	} else if( in->shape[ofastdim] <= 16 ) { // TODO: Tune this heuristic
		return transpose_vector_write(in, out, axes_actual);
	} else if( in->shape[ifastdim] <= 16 ) { // TODO: Tune this heuristic
		return transpose_vector_read(in, out, axes_actual);
	}
	
	switch( element_size ) {
#define DEFINE_TYPE_CASE(N)	  \
	case N: return typed::transpose(ndim, \
	                                in->shape, \
	                                axes_actual, \
	                                (type_of_size<N>*)in->data, \
	                                in->strides, \
	                                (type_of_size<N>*)out->data, \
	                                out->strides, \
	                                g_cuda_stream);
		DEFINE_TYPE_CASE( 1); DEFINE_TYPE_CASE( 2);
		DEFINE_TYPE_CASE( 3); DEFINE_TYPE_CASE( 4);
		DEFINE_TYPE_CASE( 5); DEFINE_TYPE_CASE( 6);
		DEFINE_TYPE_CASE( 7); DEFINE_TYPE_CASE( 8);
		DEFINE_TYPE_CASE( 9); DEFINE_TYPE_CASE(10);
		DEFINE_TYPE_CASE(11); DEFINE_TYPE_CASE(12);
		DEFINE_TYPE_CASE(13); DEFINE_TYPE_CASE(14);
		DEFINE_TYPE_CASE(15); DEFINE_TYPE_CASE(16);
#undef DEFINE_TYPE_CASE
	default: BF_FAIL("Supported bfTranspose element size",
	                 BF_STATUS_UNSUPPORTED_DTYPE);
	}
}
