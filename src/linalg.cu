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

/*
      SgemmEx: f16.f16->f16
               f16.f16->f32
                 i8.i8->f32
               f32.f32->f32
     *CgemmEx:   ci8.ci8->cf32         [>= sm_50]
               cf32.cf32->cf32         [>= sm_50]
      DgemmEx:   f64.f64->f64
      ZgemmEx: cf64.cf64->cf64
     *Cgemm3m: cf32.cf32->cf32 (Gauss) [>= sm_50]
      Zgemm3m: cf64.cf64->cf64 (Gauss) [>= sm_50]

      Cherk:   cf32->cf32
      CherkEx:  ci8->cf32              [>= sm_50]
               cf32->cf32              [>= sm_50]
     *Cherk3mEx:  ci8->cf32 (Gauss)    [>= sm_50]
                 cf32->cf32 (Gauss)    [>= sm_50]

      # TODO: Start with:
                Cgemm (+preconvert to fp32)
                CgemmEx (8bit, cuda >= 8.0, >=sm_50)
                Cgemm3m (fp32, cuda >= 8.0, >=sm_50)
                Cherk (+preconvert to fp32)
                Cherk3mEx (8bit or fp32, cuda >= 8.0, >=sm_50)
              The preconvert paths should support ci4, ci8, ci16, fp16
                The other paths should only be used if the dtype already matches
              Eventually it will probably be worth integrating the xGPU kernel,
                given the lack of cublasHerkEx (particularly the small-N problem).
*/

#include <bifrost/linalg.h>
#include "linalg_kernels.h"
#include "assert.hpp"
#include "utils.hpp"
#include "cuda.hpp"
#include "cuda/stream.hpp"
#include "ShapeIndexer.cuh"
#include "trace.hpp"

class BFlinalg_impl {
	cublasHandle_t _cublas;
	// No copy-assign
	BFlinalg_impl(BFlinalg_impl const& );
	BFlinalg_impl& operator=(BFlinalg_impl const& );
public:
	BFlinalg_impl() {
		BF_CHECK_CUBLAS_EXCEPTION(cublasCreate(&_cublas));
	}
	~BFlinalg_impl() {
		if( _cublas ) {
			cublasDestroy(_cublas);
		}
	}
	cublasHandle_t cublas() const { return _cublas; }
};

BFstatus bfMatMul_aa_exec(BFlinalg    handle,
                          cudaStream_t stream,
                          cublasOperation_t trans,
                          long        n,
                          long        k,
                          double      alpha,
                          void const* a_data,
                          BFdtype     a_type,
                          long        a_stride,
                          double      beta,
                          void const* c_data,
                          BFdtype     c_type,
                          long        c_stride) {
	BF_TRACE_STREAM(stream);
	BF_CHECK_CUBLAS(cublasSetStream(handle->cublas(), stream));
	// Note: UPPER here means lower for row-major ordering
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
	BF_CHECK_CUBLAS(cublasSetPointerMode(handle->cublas(),
	                                     CUBLAS_POINTER_MODE_HOST));
	BF_ASSERT(a_data, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(c_data, BF_STATUS_INVALID_POINTER);
	switch( a_type ) {
	case BF_DTYPE_F32: {
		BF_ASSERT(c_type == BF_DTYPE_F32, BF_STATUS_UNSUPPORTED_DTYPE);
		float alpha_f = (float)alpha;
		float beta_f  = (float)beta;
		BF_CHECK_CUBLAS(cublasSsyrk(handle->cublas(), uplo, trans,
		                            n, k,
		                            &alpha_f,
		                            (float*)a_data, a_stride,
		                            &beta_f,
		                            (float*)c_data, c_stride));
		break;
	}
	case BF_DTYPE_F64: {
		BF_ASSERT(c_type == BF_DTYPE_F64, BF_STATUS_UNSUPPORTED_DTYPE);
		BF_CHECK_CUBLAS(cublasDsyrk(handle->cublas(), uplo, trans,
		                            n, k,
		                            &alpha,
		                            (double*)a_data, a_stride,
		                            &beta,
		                            (double*)c_data, c_stride));
		break;
	}
#if CUDART_VERSION >= 8000
	case BF_DTYPE_CI8: {
		BF_ASSERT(c_type == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
		float alpha_f = (float)alpha;
		float beta_f  = (float)beta;
		if( get_cuda_device_cc() >= 50 ) {
			BF_CHECK_CUBLAS(cublasCherk3mEx(handle->cublas(), uplo, trans,
			                                n, k,
			                                &alpha_f,
			                                (cuComplex*)a_data,
			                                CUDA_C_8I,
			                                a_stride,
			                                &beta_f,
			                                (cuComplex*)c_data,
			                                CUDA_C_32F,
			                                c_stride));
			break;
		}
		BF_FAIL("Supported dtype for array a", BF_STATUS_UNSUPPORTED_DTYPE);
	}
#endif
	case BF_DTYPE_CF32: {
		BF_ASSERT(c_type == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
		float alpha_f = (float)alpha;
		float beta_f  = (float)beta;
#if CUDART_VERSION >= 8000
		if( get_cuda_device_cc() >= 50 ) {
			BF_CHECK_CUBLAS(cublasCherk3mEx(handle->cublas(), uplo, trans,
			                                n, k,
			                                &alpha_f,
			                                (cuComplex*)a_data,
			                                CUDA_C_32F,
			                                a_stride,
			                                &beta_f,
			                                (cuComplex*)c_data,
			                                CUDA_C_32F,
			                                c_stride));
			break;
		}
#endif
		BF_CHECK_CUBLAS(cublasCherk(handle->cublas(), uplo, trans,
		                            n, k,
		                            &alpha_f,
		                            (cuComplex*)a_data, a_stride,
		                            &beta_f,
		                            (cuComplex*)c_data, c_stride));
		break;
	}
	case BF_DTYPE_CF64: {
		BF_ASSERT(c_type == BF_DTYPE_CF64, BF_STATUS_UNSUPPORTED_DTYPE);
		BF_CHECK_CUBLAS(cublasZherk(handle->cublas(), uplo, trans,
		                            n, k,
		                            &alpha,
		                            (cuDoubleComplex*)a_data, a_stride,
		                            &beta,
		                            (cuDoubleComplex*)c_data, c_stride));
		break;
	}
	default:
		BF_FAIL("Supported dtype for array a", BF_STATUS_UNSUPPORTED_DTYPE);
	}
	return BF_STATUS_SUCCESS;
}

BFstatus bfMatMul_aa(BFlinalg       handle,
                     double         alpha,
                     BFarray const* a,
                     double         beta,
                     BFarray const* c) {
	BF_TRACE();
	BF_ASSERT(c->ndim == a->ndim, BF_STATUS_INVALID_SHAPE);
	int ndim = a->ndim;
	// Check that output shape is correct
	BF_ASSERT(c->shape[ndim-1] == a->shape[ndim-2], BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(c->shape[ndim-2] == a->shape[ndim-2], BF_STATUS_INVALID_SHAPE);
	// TODO: Need to check for matching batch dims? Does broadcasting work?
	// Convert byte strides to element strides
	int batch_shape[BF_MAX_DIMS];
	int astrides[BF_MAX_DIMS];
	int cstrides[BF_MAX_DIMS];
	for( int d=0; d<ndim ; ++d ) {
		batch_shape[d] = a->shape[d];
		astrides[d] = a->strides[d];
		cstrides[d] = c->strides[d];
	}
	for( int d=0; d<ndim ; ++d ) {
		BF_ASSERT(astrides[d] % BF_DTYPE_NBYTE(a->dtype) == 0, BF_STATUS_INVALID_STRIDE);
		BF_ASSERT(cstrides[d] % BF_DTYPE_NBYTE(c->dtype) == 0, BF_STATUS_INVALID_STRIDE);
		astrides[d] /= BF_DTYPE_NBYTE(a->dtype);
		cstrides[d] /= BF_DTYPE_NBYTE(c->dtype);
	}
	// Determine transposition based on strides, and update strides
	cublasOperation_t trans;
	if( astrides[ndim-1] < astrides[ndim-2] ) {
		// Note: The fastest dim cannot be a batch dim
		BF_ASSERT(astrides[ndim-1] == 1, BF_STATUS_UNSUPPORTED_STRIDE);
		trans = (BF_DTYPE_IS_COMPLEX(a->dtype) ?
		         CUBLAS_OP_C :
		         CUBLAS_OP_T);
	} else if( astrides[ndim-1] > astrides[ndim-2] ) {
		// Note: The fastest dim cannot be a batch dim
		BF_ASSERT(astrides[ndim-2] == 1, BF_STATUS_UNSUPPORTED_STRIDE);
		trans = CUBLAS_OP_N;
		std::swap(astrides[ndim-1], astrides[ndim-2]);
	} else {
		BF_ASSERT(false, BF_STATUS_INVALID_STRIDE);
	}
	BF_ASSERT(cstrides[ndim-2] > cstrides[ndim-1], BF_STATUS_UNSUPPORTED_STRIDE);
	// Loop over batch dims
	ShapeIndexer<BF_MAX_DIMS> shape_indexer(batch_shape, ndim-2);
	for( long i=0; i<shape_indexer.size(); ++i ) {
		auto inds = shape_indexer.at(i);
		void* a_data = array_get_pointer(a, inds);
		void* c_data = array_get_pointer(c, inds);
		cuda::child_stream stream(g_cuda_stream);
		BF_CHECK( bfMatMul_aa_exec(handle, stream, trans,
		                           a->shape[ndim-2], a->shape[ndim-1],
		                           alpha, a_data, a->dtype, astrides[ndim-2],
		                           beta,  c_data, c->dtype, cstrides[ndim-2]) );
	}
	return BF_STATUS_SUCCESS;
}

BFstatus bfMatMul_ab_exec(BFlinalg    handle,
                          cudaStream_t stream,
                          cublasOperation_t trans_a,
                          cublasOperation_t trans_b,
                          long        m,
                          long        n,
                          long        k,
                          double      alpha,
                          void const* a_data,
                          BFdtype     a_type,
                          long        a_stride,
                          void const* b_data,
                          BFdtype     b_type,
                          long        b_stride,
                          double      beta,
                          void const* c_data,
                          BFdtype     c_type,
                          long        c_stride) {
	BF_TRACE_STREAM(stream);
	BF_CHECK_CUBLAS(cublasSetStream(handle->cublas(), stream));
	BF_CHECK_CUBLAS(cublasSetPointerMode(handle->cublas(),
	                                     CUBLAS_POINTER_MODE_HOST));
	BF_ASSERT(a_data, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(b_data, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(c_data, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(a_type == b_type, BF_STATUS_UNSUPPORTED_DTYPE);
	// TODO: Look into optimizations using cublasGemmEx algo selection and
	//         batched/strided APIs.
	switch( a_type ) {
	case BF_DTYPE_F32: {
		BF_ASSERT(c_type == BF_DTYPE_F32, BF_STATUS_UNSUPPORTED_DTYPE);
		float alpha_f = (float)alpha;
		float beta_f  = (float)beta;
		BF_CHECK_CUBLAS(cublasSgemm(handle->cublas(), trans_a, trans_b,
		                            m, n, k,
		                            &alpha_f,
		                            (float*)a_data, a_stride,
		                            (float*)b_data, b_stride,
		                            &beta_f,
		                            (float*)c_data, c_stride));
		break;
	}
	case BF_DTYPE_F64: {
		BF_ASSERT(c_type == BF_DTYPE_F64, BF_STATUS_UNSUPPORTED_DTYPE);
		BF_CHECK_CUBLAS(cublasDgemm(handle->cublas(), trans_a, trans_b,
		                            m, n, k,
		                            &alpha,
		                            (double*)a_data, a_stride,
		                            (double*)b_data, b_stride,
		                            &beta,
		                            (double*)c_data, c_stride));
		break;
	}
#if CUDART_VERSION >= 8000
	case BF_DTYPE_CI8: {
		BF_ASSERT(c_type == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
		cuComplex alpha_cf = make_cuComplex(alpha, 0);
		cuComplex beta_cf  = make_cuComplex(beta,  0);
		if( get_cuda_device_cc() >= 50 ) {
			BF_CHECK_CUBLAS(cublasCgemmEx(handle->cublas(), trans_a, trans_b,
			                              m, n, k,
			                              &alpha_cf,
			                              (cuComplex*)a_data,
			                              CUDA_C_8I,
			                              a_stride,
			                              (cuComplex*)b_data,
			                              CUDA_C_8I,
			                              b_stride,
			                              &beta_cf,
			                              (cuComplex*)c_data,
			                              CUDA_C_32F,
			                              c_stride));
			break;
		}
		BF_FAIL("Supported dtype for input array", BF_STATUS_UNSUPPORTED_DTYPE);
	}
#endif
	case BF_DTYPE_CF32: {
		BF_ASSERT(c_type == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
		cuComplex alpha_cf = make_cuComplex(alpha, 0);
		cuComplex beta_cf  = make_cuComplex(beta,  0);
#if CUDART_VERSION >= 8000
		if( get_cuda_device_cc() >= 50 ) {
			BF_CHECK_CUBLAS(cublasCgemm3m(handle->cublas(), trans_a, trans_b,
			                              m, n, k,
			                              &alpha_cf,
			                              (cuComplex*)a_data,
			                              a_stride,
			                              (cuComplex*)b_data,
			                              b_stride,
			                              &beta_cf,
			                              (cuComplex*)c_data,
			                              c_stride));
			break;
		}
#endif
		BF_CHECK_CUBLAS(cublasCgemm(handle->cublas(), trans_a, trans_b,
		                            m, n, k,
		                            &alpha_cf,
		                            (cuComplex*)a_data, a_stride,
		                            (cuComplex*)b_data, b_stride,
		                            &beta_cf,
		                            (cuComplex*)c_data, c_stride));
		break;
	}
	case BF_DTYPE_CF64: {
		cuDoubleComplex alpha_cd = make_cuDoubleComplex(alpha, 0);
		cuDoubleComplex beta_cd  = make_cuDoubleComplex(beta,  0);
		BF_ASSERT(c_type == BF_DTYPE_CF64, BF_STATUS_UNSUPPORTED_DTYPE);
		BF_CHECK_CUBLAS(cublasZgemm(handle->cublas(), trans_a, trans_b,
		                            m, n, k,
		                            &alpha_cd,
		                            (cuDoubleComplex*)a_data, a_stride,
		                            (cuDoubleComplex*)b_data, b_stride,
		                            &beta_cd,
		                            (cuDoubleComplex*)c_data, c_stride));
		break;
	}
	default:
		BF_FAIL("Supported dtype for input array", BF_STATUS_UNSUPPORTED_DTYPE);
	}
	return BF_STATUS_SUCCESS;
}

BFstatus bfMatMul_ab(BFlinalg       handle,
                     double         alpha,
                     BFarray const* a,
                     BFarray const* b,
                     double         beta,
                     BFarray const* c) {
	BF_TRACE();
	BF_ASSERT(c->ndim == a->ndim, BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(c->ndim == b->ndim, BF_STATUS_INVALID_SHAPE);
	int ndim = a->ndim;
	// Check that shapes are correct
	BF_ASSERT(c->shape[ndim-2] == a->shape[ndim-2], BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(c->shape[ndim-1] == b->shape[ndim-1], BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(a->shape[ndim-1] == b->shape[ndim-2], BF_STATUS_INVALID_SHAPE);
	// TODO: Need to check for matching batch dims? Does broadcasting work?
	// Convert byte strides to element strides
	int batch_shape[BF_MAX_DIMS];
	int astrides[BF_MAX_DIMS];
	int bstrides[BF_MAX_DIMS];
	int cstrides[BF_MAX_DIMS];
	for( int d=0; d<ndim ; ++d ) {
		batch_shape[d] = a->shape[d];
		astrides[d] = a->strides[d];
		bstrides[d] = b->strides[d];
		cstrides[d] = c->strides[d];
	}
	for( int d=0; d<ndim ; ++d ) {
		BF_ASSERT(astrides[d] % BF_DTYPE_NBYTE(a->dtype) == 0, BF_STATUS_INVALID_STRIDE);
		BF_ASSERT(bstrides[d] % BF_DTYPE_NBYTE(b->dtype) == 0, BF_STATUS_INVALID_STRIDE);
		BF_ASSERT(cstrides[d] % BF_DTYPE_NBYTE(c->dtype) == 0, BF_STATUS_INVALID_STRIDE);
		astrides[d] /= BF_DTYPE_NBYTE(a->dtype);
		bstrides[d] /= BF_DTYPE_NBYTE(b->dtype);
		cstrides[d] /= BF_DTYPE_NBYTE(c->dtype);
	}
	// Determine transposition based on strides, and update strides
	cublasOperation_t trans_a;
	cublasOperation_t trans_b;
	if( astrides[ndim-1] < astrides[ndim-2] ) {
		// Note: The fastest dim cannot be a batch dim
		BF_ASSERT(astrides[ndim-1] == 1, BF_STATUS_UNSUPPORTED_STRIDE);
		// TODO: Check behaviour with conjugated arrays
		BF_ASSERT(!BF_DTYPE_IS_COMPLEX(a->dtype) || !a->conjugated,
		          BF_STATUS_UNSUPPORTED);
		trans_a = CUBLAS_OP_N;
	} else if( astrides[ndim-1] > astrides[ndim-2] ) {
		// Note: The fastest dim cannot be a batch dim
		BF_ASSERT(astrides[ndim-2] == 1, BF_STATUS_UNSUPPORTED_STRIDE);
		trans_a = (BF_DTYPE_IS_COMPLEX(a->dtype) && a->conjugated ?
		           CUBLAS_OP_C :
		           CUBLAS_OP_T);
		std::swap(astrides[ndim-1], astrides[ndim-2]);
	} else {
		BF_ASSERT(false, BF_STATUS_INVALID_STRIDE);
	}
	if( bstrides[ndim-1] < bstrides[ndim-2] ) {
		// Note: The fastest dim cannot be a batch dim
		BF_ASSERT(bstrides[ndim-1] == 1, BF_STATUS_UNSUPPORTED_STRIDE);
		// TODO: Check behaviour with conjugated arrays
		BF_ASSERT(!BF_DTYPE_IS_COMPLEX(b->dtype) || !b->conjugated,
		          BF_STATUS_UNSUPPORTED);
		trans_b = CUBLAS_OP_N;
	} else if( bstrides[ndim-1] > bstrides[ndim-2] ) {
		// Note: The fastest dim cannot be a batch dim
		BF_ASSERT(bstrides[ndim-2] == 1, BF_STATUS_UNSUPPORTED_STRIDE);
		trans_b = (BF_DTYPE_IS_COMPLEX(b->dtype) && b->conjugated ?
		           CUBLAS_OP_C :
		           CUBLAS_OP_T);
		std::swap(bstrides[ndim-1], bstrides[ndim-2]);
	} else {
		BF_ASSERT(false, BF_STATUS_INVALID_STRIDE);
	}
	BF_ASSERT(cstrides[ndim-2] > cstrides[ndim-1], BF_STATUS_UNSUPPORTED_STRIDE);
	// Loop over batch dims
	ShapeIndexer<BF_MAX_DIMS> shape_indexer(batch_shape, ndim-2);
	for( long i=0; i<shape_indexer.size(); ++i ) {
		auto inds = shape_indexer.at(i);
		void* a_data = array_get_pointer(a, inds);
		void* b_data = array_get_pointer(b, inds);
		void* c_data = array_get_pointer(c, inds);
		cuda::child_stream stream(g_cuda_stream);
		BF_CHECK( bfMatMul_ab_exec(handle, stream,  trans_b,trans_a,
		                           c->shape[ndim-1], // m
		                           c->shape[ndim-2], // n
		                           a->shape[ndim-1], // k
		                           alpha,
		                           // Note: We swap a and b here because
		                           //         CUBLAS uses column-major
		                           //         while we use row-major order.
		                           b_data, b->dtype, bstrides[ndim-2],
		                           a_data, a->dtype, astrides[ndim-2],
		                           beta,
		                           c_data, c->dtype, cstrides[ndim-2]) );
	}
	return BF_STATUS_SUCCESS;
}

BFstatus bfLinAlgCreate(BFlinalg* handle_ptr) {
	BF_TRACE();
	BF_ASSERT(handle_ptr, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*handle_ptr = new BFlinalg_impl(),
	                   *handle_ptr = 0);
}
BFstatus bfLinAlgDestroy(BFlinalg handle) {
	BF_TRACE();
	BF_ASSERT(handle, BF_STATUS_INVALID_HANDLE);
	delete handle;
	return BF_STATUS_SUCCESS;
}
// Computes c = a.b or a.a^H if b is NULL
BFstatus bfLinAlgMatMul(BFlinalg       handle,
                        double         alpha,
                        BFarray const* a,   // [...,i,j]
                        BFarray const* b,   // [...,j,k]
                        double         beta,
                        BFarray const* c) {  // [...,i,k]
	// TODO: Use weight_and_sum kernel when:
	//         Dim i is the fastest dim of a
	//         Dim j is the fastest dim of b
	//         Dim k is NOT the fastest dim of c
	//         [Dim k is small (say < 64)]
	// TODO: Generalise weight_and_sum kernel to arbitrary strides and dtypes
	//         For dtypes, need Complex<T> to work for vectorized loads
	//           UNLESS, we use something like storage_type<T>::type
	BF_TRACE();
	BF_ASSERT(handle, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(a, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(c, BF_STATUS_INVALID_POINTER);
	if( b ) {
		return bfMatMul_ab(handle, alpha, a, b, beta, c);
	} else {
		return bfMatMul_aa(handle, alpha, a, beta, c);
	}
}
