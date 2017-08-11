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

// *TODO: Implement specialised kernel for reducing > 128-bit elements over the
//          fastest-changing axis. Currently these cases run 10x slower than
//          other cases.
//          Consider using the beamformer kernel approach of using looped warp
//            reductions.

#include <bifrost/reduce.h>
#include "assert.hpp"
#include "utils.hpp"
#include "cuda/stream.hpp"

template<typename T, int N>
struct __attribute__((aligned(sizeof(T)*N))) aligned_vector_type {
	T v[N];
};

// A specialized kernel for reducing over a small (<= 128-bit) fastest-changing
//   dimension. Each thread performs a vector load and reduces the elements
//   sequentially.
template<int N, typename IType, typename OType>
__global__
void reduce_vector_kernel(IType const* __restrict__ in,
                          OType*       __restrict__ out,
                          int4 shape,
                          int4 istrides,
                          int4 ostrides,
                          BFreduce_op op) {
	int4 i0 = make_int4(threadIdx.x + blockIdx.x * blockDim.x,
	                    threadIdx.y + blockIdx.y * blockDim.y,
	                    threadIdx.z + blockIdx.z * blockDim.z,
	                    0);
	int4 i;
	for( i.z=i0.z; i.z<shape.z; i.z+=blockDim.z*gridDim.z ) {
		for( i.y=i0.y; i.y<shape.y; i.y+=blockDim.y*gridDim.y ) {
			for( i.x=i0.x; i.x<shape.x; i.x+=blockDim.x*gridDim.x ) {
				int iidx0 = (i.x * istrides.x +
				             i.y * istrides.y +
				             i.z * istrides.z);
				typedef aligned_vector_type<IType, N> VType;
				VType ivals = type_pun<VType const*>(in)[iidx0 / N];
				OType result = ivals.v[0];
#pragma unroll
				for( int n=1; n<N; ++n ) {
					OType ival = (OType)ivals.v[n];
					switch( op ) {
					case BF_REDUCE_SUM:  // Fall-through
					case BF_REDUCE_MEAN: // Fall-through
					case BF_REDUCE_STDERR: result += ival; break;
					case BF_REDUCE_MIN:    result = min(result, ival); break;
					case BF_REDUCE_MAX:    result = max(result, ival); break;
					}
				}
				switch( op ) {
				case BF_REDUCE_MEAN:   result *= 1. / N; break;
				case BF_REDUCE_STDERR: result *= 1. / sqrtf(N); break;
				}
				int oidx = (i.x * ostrides.x +
				            i.y * ostrides.y +
				            i.z * ostrides.z);
				out[oidx] = result;
			}
		}
	}
}

template<int N, typename IType, typename OType>
void launch_reduce_vector_kernel(IType const* in,
                                 OType*       out,
                                 int4 shape,
                                 int4 istrides,
                                 int4 ostrides,
                                 BFreduce_op op,
                                 cudaStream_t stream) {
	dim3 block(128); // TODO: Tune this
	dim3 grid(std::min((shape.x - 1) / block.x + 1, 65535u),
	          std::min(shape.y, 65535),
	          std::min(shape.z, 65535));
	void* args[] = {
		&in,
		&out,
		&shape,
		&istrides,
		&ostrides,
		&op
	};
	BF_CHECK_CUDA_EXCEPTION(
		cudaLaunchKernel((void*)reduce_vector_kernel<N, IType, OType>,
		                 grid, block, &args[0], 0, stream),
		BF_STATUS_INTERNAL_ERROR);
}

// TODO: Allow processing multiple elements per thread (with vector ld/st)
//         to improve perf of 8- and 16-bit input (which is currently 2-3x
//         slower than 32-bit loads).
//         Should use 2x or 4x elements per thread whenever shape and strides
//           are aligned to these amounts.
// A simple kernel designed for reducing over non-fastest dimensions
//   Each thread loops sequentially over the reduced dimension
template<typename IType, typename OType>
__global__
void reduce_loop_kernel(IType const* __restrict__ in,
                        OType*       __restrict__ out,
                        int4 shape,
                        int4 istrides,
                        int4 ostrides,
                        BFreduce_op op) {
	int4 i0 = make_int4(threadIdx.x + blockIdx.x * blockDim.x,
	                    threadIdx.y + blockIdx.y * blockDim.y,
	                    threadIdx.z + blockIdx.z * blockDim.z,
	                    0);
	int4 i;
	for( i.z=i0.z; i.z<shape.z; i.z+=blockDim.z*gridDim.z ) {
		for( i.y=i0.y; i.y<shape.y; i.y+=blockDim.y*gridDim.y ) {
			for( i.x=i0.x; i.x<shape.x; i.x+=blockDim.x*gridDim.x ) {
				int iidx0 = (i.x * istrides.x +
				             i.y * istrides.y +
				             i.z * istrides.z);
				OType result = (OType)in[iidx0];
				for( i.w=1; i.w<shape.w; ++i.w ) {
					OType ival = (OType)in[iidx0 + i.w * istrides.w];
					switch( op ) {
					case BF_REDUCE_SUM:  // Fall-through
					case BF_REDUCE_MEAN: // Fall-through
					case BF_REDUCE_STDERR: result += ival; break;
					case BF_REDUCE_MIN:    result = min(result, ival); break;
					case BF_REDUCE_MAX:    result = max(result, ival); break;
					}
				}
				switch( op ) {
				case BF_REDUCE_MEAN:   result *= 1. / shape.w; break;
				case BF_REDUCE_STDERR: result *= 1. / sqrtf(shape.w); break;
				}
				int oidx = (i.x * ostrides.x +
				            i.y * ostrides.y +
				            i.z * ostrides.z);
				out[oidx] = result;
			}
		}
	}
}

template<typename IType, typename OType>
void launch_reduce_loop_kernel(IType const* in,
                               OType*       out,
                               int4 shape,
                               int4 istrides,
                               int4 ostrides,
                               BFreduce_op op,
                               cudaStream_t stream) {
	dim3 block(128);
	if( istrides.w == 1 && shape.w > 16 ) {
		// This gives better perf when reducing along the fastest-changing axis
		//   (non-coalesced reads).
		block.x = 16;
		block.y = 16;
	}
	dim3 grid(std::min((shape.x - 1) / block.x + 1, 65535u),
	          std::min((shape.y - 1) / block.y + 1, 65535u),
	          std::min(shape.z, 65535));
	void* args[] = {
		&in,
		&out,
		&shape,
		&istrides,
		&ostrides,
		&op
	};
	BF_CHECK_CUDA_EXCEPTION(
		cudaLaunchKernel((void*)reduce_loop_kernel<IType, OType>,
		                 grid, block, &args[0], 0, stream),
		BF_STATUS_INTERNAL_ERROR);
}

template<typename T>
T signed_left_shift(T x, int shift) {
	if( shift >= 0 ) {
		return x << shift;
	} else {
		return x >> -shift;
	}
}

template<typename IType, typename OType>
BFstatus reduce_itype_otype(BFarray const* in,
                                 BFarray const* out,
                                 BFreduce_op    op,
                                 int            axis) {
	BF_ASSERT(in->shape[axis] % out->shape[axis] == 0, BF_STATUS_INVALID_SHAPE);
	long reduce_size    =  in->shape[axis] / out->shape[axis];
	long istride_reduce =  in->strides[axis];
	BFarray out_flattened, in_flattened;
	split_dim(in, &in_flattened, axis, reduce_size);
	remove_dim(&in_flattened, &in_flattened, axis + 1);
	unsigned long keep_dims_mask = 0;
	keep_dims_mask |= padded_dims_mask(&in_flattened);
	keep_dims_mask |= padded_dims_mask(out);
	flatten( &in_flattened,  &in_flattened, keep_dims_mask);
	flatten(           out, &out_flattened, keep_dims_mask);
	in  =  &in_flattened;
	out = &out_flattened;
	BF_ASSERT(in_flattened.ndim == out_flattened.ndim,
	          BF_STATUS_INTERNAL_ERROR);
	int ndim = in_flattened.ndim;
	BF_ASSERT(ndim <= 4, BF_STATUS_UNSUPPORTED_SHAPE);
	
	BF_ASSERT(istride_reduce % BF_DTYPE_NBYTE( in->dtype) == 0, BF_STATUS_UNSUPPORTED_STRIDE);
	istride_reduce /= BF_DTYPE_NBYTE( in->dtype);
	for( int d=0; d<ndim; ++d ) {
		BF_ASSERT( in->strides[d] % BF_DTYPE_NBYTE( in->dtype) == 0, BF_STATUS_UNSUPPORTED_STRIDE);
		BF_ASSERT(out->strides[d] % BF_DTYPE_NBYTE(out->dtype) == 0, BF_STATUS_UNSUPPORTED_STRIDE);
		 in_flattened.strides[d] /= BF_DTYPE_NBYTE( in->dtype);
		out_flattened.strides[d] /= BF_DTYPE_NBYTE(out->dtype);
	}
	
	int4 shape    = make_int4(ndim > 0 ? out->shape[ndim-1-0] : 1,
	                          ndim > 1 ? out->shape[ndim-1-1] : 1,
	                          ndim > 2 ? out->shape[ndim-1-2] : 1,
	                          reduce_size);
	int4 istrides = make_int4(ndim > 0 ? in->strides[ndim-1-0] : 0,
	                          ndim > 1 ? in->strides[ndim-1-1] : 0,
	                          ndim > 2 ? in->strides[ndim-1-2] : 0,
	                          istride_reduce);
	int4 ostrides = make_int4(ndim > 0 ? out->strides[ndim-1-0] : 0,
	                          ndim > 1 ? out->strides[ndim-1-1] : 0,
	                          ndim > 2 ? out->strides[ndim-1-2] : 0,
	                          0);
	bool use_vec2_kernel = (
		istride_reduce == 1 &&
		reduce_size == 2 &&
		istrides.x % 2 == 0 && istrides.y % 2 == 0 && istrides.z % 2 == 0);
	//std::cout << istride_reduce << ", " << reduce_size << ", " << istrides.x << ", " << istrides.y << ", " << istrides.z << std::endl;
	bool use_vec4_kernel = (
		istride_reduce == 1 &&
		reduce_size == 4 &&
		istrides.x % 4 == 0 && istrides.y % 4 == 0 && istrides.z % 4 == 0);
	bool use_vec8_kernel = (
		sizeof(IType) <= 2 &&
		istride_reduce == 1 &&
		reduce_size == 8 &&
		istrides.x % 8 == 0 && istrides.y % 8 == 0 && istrides.z % 8 == 0);
	bool use_vec16_kernel = (
		sizeof(IType) == 1 &&
		istride_reduce == 1 &&
		reduce_size == 16 &&
		istrides.x % 16 == 0 && istrides.y % 16 == 0 && istrides.z % 16 == 0);
	
	if( use_vec2_kernel ) {
		BF_TRY_RETURN(
			launch_reduce_vector_kernel<2>((IType*)in->data, (OType*)out->data,
			                               shape, istrides, ostrides,
			                               op, g_cuda_stream));
	} else if( use_vec4_kernel ) {
		BF_TRY_RETURN(
			launch_reduce_vector_kernel<4>((IType*)in->data, (OType*)out->data,
			                               shape, istrides, ostrides,
			                               op, g_cuda_stream));
	} else if( use_vec8_kernel ) {
		BF_TRY_RETURN(
			launch_reduce_vector_kernel<8>((IType*)in->data, (OType*)out->data,
			                               shape, istrides, ostrides,
			                               op, g_cuda_stream));
	} else if( use_vec16_kernel ) {
		BF_TRY_RETURN(
			launch_reduce_vector_kernel<16>((IType*)in->data, (OType*)out->data,
			                               shape, istrides, ostrides,
			                               op, g_cuda_stream));
	} else {
		BF_TRY_RETURN(
			launch_reduce_loop_kernel((IType*)in->data, (OType*)out->data,
			                          shape, istrides, ostrides,
			                          op, g_cuda_stream));
	}
}

template<typename IType>
BFstatus reduce_itype(BFarray const* in,
                       BFarray const* out,
                       BFreduce_op    op,
                       int            axis) {
	switch( out->dtype ) {
	case BF_DTYPE_F32:
		return reduce_itype_otype<IType,float>(in, out, op, axis);
	default: BF_FAIL("Supported output dtype", BF_STATUS_UNSUPPORTED_DTYPE);
	}
}

BFstatus reduce(BFarray const* in,
                BFarray const* out,
                BFreduce_op    op,
                int            axis) {
	switch( in->dtype ) {
	case BF_DTYPE_I8:  return reduce_itype<int8_t  >(in, out, op, axis);
	case BF_DTYPE_I16: return reduce_itype<int16_t >(in, out, op, axis);
	case BF_DTYPE_U8:  return reduce_itype<uint8_t >(in, out, op, axis);
	case BF_DTYPE_U16: return reduce_itype<uint16_t>(in, out, op, axis);
	case BF_DTYPE_F32: return reduce_itype<float   >(in, out, op, axis);
	default: BF_FAIL("Supported input dtype", BF_STATUS_UNSUPPORTED_DTYPE);
	}
}

BFstatus bfReduce(BFarray const* in, BFarray const* out, BFreduce_op op) {
	BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
	BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
	BF_ASSERT(in->ndim == out->ndim, BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(space_accessible_from(in->space, BF_SPACE_CUDA),
	          BF_STATUS_UNSUPPORTED_SPACE);
	
	int ndim = in->ndim;
	int ndim_reduce = 0;
	int reduce_dims[BF_MAX_DIMS];
	for( int d=0; d<ndim; ++d ) {
		BF_ASSERT(out->shape[d] <= in->shape[d], BF_STATUS_INVALID_SHAPE);
		if( out->shape[d] < in->shape[d] ) {
			reduce_dims[ndim_reduce++] = d;
		}
	}
	BF_ASSERT(ndim_reduce  > 0, BF_STATUS_INVALID_SHAPE);
	// TODO: Eventually want to support reductions over multiple dims
	//         E.g., image downsampling
	BF_ASSERT(ndim_reduce == 1, BF_STATUS_UNSUPPORTED_SHAPE);
	
	return reduce(in, out, op, reduce_dims[0]);
}
