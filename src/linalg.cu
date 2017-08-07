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
#include "Complex.hpp"

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

BFstatus bfMatMul_aa_exec_nobatch(BFlinalg    handle,
                                  cudaStream_t stream,
                                  cublasOperation_t trans,
                                  long        n,
                                  long        k,
                                  double      alpha,
                                  void const* a_data,
                                  BFdtype     a_type,
                                  long        a_stride,
                                  double      beta,
                                  void*       c_data,
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

BFstatus bfMatMul_aa_exec(BFlinalg    handle,
                          cudaStream_t stream,
                          cublasOperation_t trans,
                          long        n,
                          long        k,
                          long        nbatch,
                          double      alpha,
                          void const* a_data,
                          BFdtype     a_type,
                          long        a_stride,
                          long        a_batchstride,
                          double      beta,
                          void*       c_data,
                          BFdtype     c_type,
                          long        c_stride,
                          long        c_batchstride) {
	// TODO: Use batched algos here where possible
	
	//char* use_bf_cherk_str = getenv("BF_CHERK");
	//bool use_bf_cherk = use_bf_cherk_str && atoi(use_bf_cherk_str);
	enum { BF_CUBLAS_CHERK_THRESHOLD = 896 };
	if( //use_bf_cherk &&
	    (CUDART_VERSION < 8000 || n < BF_CUBLAS_CHERK_THRESHOLD) &&
	    trans == CUBLAS_OP_N &&
	    k % 4 == 0 &&
	    n % 2 == 0 &&
	    a_stride % 2 == 0 && a_batchstride % 2 == 0 &&
	    c_stride % 2 == 0 && c_batchstride % 2 == 0 &&
	    (a_type == BF_DTYPE_CI8 || a_type == BF_DTYPE_CI16) &&
	    c_type == BF_DTYPE_CF32 ) {
		BF_TRY_RETURN(bf_cherk_N(
			n, k, nbatch,
			alpha,
			a_data, a_type, a_stride, a_batchstride,
			beta,
			c_data, c_type, c_stride, c_batchstride,
			stream));
	}
	
	for( long b=0; b<nbatch; ++b ) {
		cuda::child_stream child_stream(stream);
		BF_CHECK( bfMatMul_aa_exec_nobatch(handle, child_stream,
		                                   trans, n, k,
		                                   alpha,
		                                   a_data, a_type, a_stride,
		                                   beta,
		                                   c_data, c_type, c_stride) );
		a_data = (char*)a_data + a_batchstride * BF_DTYPE_NBYTE(a_type);
		c_data = (char*)c_data + c_batchstride * BF_DTYPE_NBYTE(c_type);
	}
	return BF_STATUS_SUCCESS;
}

BFstatus bfMatMul_aa(BFlinalg       handle,
                     double         alpha,
                     BFarray const* a,
                     bool           adjoint,
                     double         beta,
                     BFarray const* c) {
	BF_TRACE();
	BF_ASSERT(c->ndim == a->ndim, BF_STATUS_INVALID_SHAPE);
	int ndim = a->ndim;
	BFarray a_mutable;
	::memcpy(&a_mutable, a, sizeof(BFarray));
	a = &a_mutable;
	if( adjoint ) {
		std::swap(a_mutable.shape[  ndim-1], a_mutable.shape[  ndim-2]);
		std::swap(a_mutable.strides[ndim-1], a_mutable.strides[ndim-2]);
		a_mutable.conjugated = !a_mutable.conjugated;
	}
	// Check that output shape is correct
	BF_ASSERT(c->shape[ndim-1] == a->shape[ndim-2], BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(c->shape[ndim-2] == a->shape[ndim-2], BF_STATUS_INVALID_SHAPE);
	
	// Handle batch dims by merging the contiguous ones together and selecting
	//   the largest one to be the kernel batch dim.
	long nbatch = 1;
	int  batch_dim = -1;
	int  batch_shape[BF_MAX_DIMS];
	BFarray a_flattened, c_flattened;
	for( int d=0; d<ndim; ++d ) {
		batch_shape[d] = 1;
	}
	if( ndim > 2 ) {
		// Keep the last 3 dims but attempt to flatten all others
		unsigned long keep_dims_mask = 0x7 << (ndim-3);
		keep_dims_mask |= padded_dims_mask(a);
		keep_dims_mask |= padded_dims_mask(c);
		flatten(a, &a_flattened, keep_dims_mask);
		flatten(c, &c_flattened, keep_dims_mask);
		a = &a_flattened;
		c = &c_flattened;
		BF_ASSERT(a_flattened.ndim == c_flattened.ndim, BF_STATUS_INTERNAL_ERROR);
		ndim = c->ndim;
		
		for( int d=0; d<ndim-2; ++d ) {
			BF_ASSERT(a->shape[d] == c->shape[d] || a->shape[d] == 1, BF_STATUS_INVALID_SHAPE);
			batch_shape[d] = c->shape[d];
			// Find longest dimension to use as kernel batch dim
			if( c->shape[d] >= nbatch ) {
				nbatch = c->shape[d];
				batch_dim = d;
			}
		}
		// Remove the kernel batch dim from the rest of the batch shape
		batch_shape[batch_dim] = 1;
	}
	
	// Convert byte strides to element strides
	int astrides[BF_MAX_DIMS];
	int cstrides[BF_MAX_DIMS];
	for( int d=0; d<ndim ; ++d ) {
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
		if( BF_DTYPE_IS_COMPLEX(a->dtype) ) {
			// Note: Because BLAS uses col-major ordering, we can only support
			//         the non-conjugated case here.
			BF_ASSERT(!a->conjugated, BF_STATUS_UNSUPPORTED);
			trans = CUBLAS_OP_C;
		} else {
			trans = CUBLAS_OP_T;
		}
	} else if( astrides[ndim-1] > astrides[ndim-2] ) {
		// Note: The fastest dim cannot be a batch dim
		BF_ASSERT(astrides[ndim-2] == 1, BF_STATUS_UNSUPPORTED_STRIDE);
		// Note: Because BLAS uses col-major ordering, we can only support
		//         the conjugated case here.
		if( BF_DTYPE_IS_COMPLEX(a->dtype) ) {
			BF_ASSERT(a->conjugated, BF_STATUS_UNSUPPORTED);
		}
		trans = CUBLAS_OP_N;
		std::swap(astrides[ndim-1], astrides[ndim-2]);
	} else {
		// TODO: I think this actually occurs legitimately when shape[-1] = 1
		BF_ASSERT(false, BF_STATUS_INVALID_STRIDE);
	}
	BF_ASSERT(cstrides[ndim-2] >= cstrides[ndim-1], BF_STATUS_UNSUPPORTED_STRIDE);
	
	if( nbatch > 1 ) {
		// Enable broadcasting in the kernel batch dim
		if( a->shape[batch_dim] == 1 ) { astrides[batch_dim] = 0; }
	}
	
	// Loop over batch dims
	ShapeIndexer<BF_MAX_DIMS> shape_indexer(batch_shape, ndim);
	for( long i=0; i<shape_indexer.size(); ++i ) {
		auto inds = shape_indexer.at(i);
		void* a_data = array_get_pointer(a, inds);
		void* c_data = array_get_pointer(c, inds);
		cuda::child_stream stream(g_cuda_stream);
		BF_CHECK( bfMatMul_aa_exec(handle, stream, trans,
		                           a->shape[ndim-2], a->shape[ndim-1], nbatch,
		                           alpha, a_data, a->dtype, astrides[ndim-2], astrides[batch_dim],
		                           beta,  c_data, c->dtype, cstrides[ndim-2], cstrides[batch_dim]) );
	}
	return BF_STATUS_SUCCESS;
}

BFstatus bfMatMul_ab_exec_nobatch(BFlinalg    handle,
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
                                  void*       c_data,
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

BFstatus bfMatMul_ab_exec(BFlinalg    handle,
                          cudaStream_t stream,
                          cublasOperation_t trans_a,
                          cublasOperation_t trans_b,
                          long        m,
                          long        n,
                          long        k,
                          long        nbatch,
                          double      alpha,
                          void const* a_data,
                          BFdtype     a_type,
                          long        a_stride,
                          long        a_batchstride,
                          void const* b_data,
                          BFdtype     b_type,
                          long        b_stride,
                          long        b_batchstride,
                          double      beta,
                          void*       c_data,
                          BFdtype     c_type,
                          long        c_stride,
                          long        c_batchstride) {
	// TODO: Use batched algos here where possible
	
	//char* use_bf_cgemm_str = getenv("BF_CGEMM");
	//bool use_bf_cgemm = use_bf_cgemm_str && atoi(use_bf_cgemm_str);
	if( //use_bf_cgemm &&
	    n <= 12 &&
	    trans_a == CUBLAS_OP_T && trans_b == CUBLAS_OP_N &&
	    (a_type == BF_DTYPE_CI4  || a_type == BF_DTYPE_CI8) &&
	    (b_type == BF_DTYPE_CI16 || b_type == BF_DTYPE_CF16 || b_type == BF_DTYPE_CF32) &&
	    c_type == BF_DTYPE_CF32 ) {
		BF_TRY_RETURN(bf_cgemm_TN_smallM(
			m, n, k, nbatch,
			alpha,
			a_data, a_type, a_stride, a_batchstride,
			b_data, b_type, b_stride, b_batchstride,
			beta,
			c_data, c_type, c_stride, c_batchstride,
			stream));
	}
	
	for( long b=0; b<nbatch; ++b ) {
		cuda::child_stream child_stream(stream);
		BF_CHECK( bfMatMul_ab_exec_nobatch(handle, child_stream,
		                                   trans_a, trans_b,
		                                   m, n, k,
		                                   alpha,
		                                   a_data, a_type, a_stride,
		                                   b_data, b_type, b_stride,
		                                   beta,
		                                   c_data, c_type, c_stride) );
		a_data = (char*)a_data + a_batchstride * BF_DTYPE_NBYTE(a_type);
		b_data = (char*)b_data + b_batchstride * BF_DTYPE_NBYTE(b_type);
		c_data = (char*)c_data + c_batchstride * BF_DTYPE_NBYTE(c_type);
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
	
	// Handle batch dims by merging the contiguous ones together and selecting
	//   the largest one to be the kernel batch dim.
	long nbatch = 1;
	int  batch_dim = -1;
	int  batch_shape[BF_MAX_DIMS];
	BFarray a_flattened, b_flattened, c_flattened;
	for( int d=0; d<ndim; ++d ) {
		batch_shape[d] = 1;
	}
	if( ndim > 2 ) {
		// Keep the last 3 dims but attempt to flatten all others
		unsigned long keep_dims_mask = 0x7 << (ndim-3);
		keep_dims_mask |= padded_dims_mask(a);
		keep_dims_mask |= padded_dims_mask(b);
		keep_dims_mask |= padded_dims_mask(c);
		flatten(a, &a_flattened, keep_dims_mask);
		flatten(b, &b_flattened, keep_dims_mask);
		flatten(c, &c_flattened, keep_dims_mask);
		a = &a_flattened;
		b = &b_flattened;
		c = &c_flattened;
		BF_ASSERT(a_flattened.ndim == b_flattened.ndim, BF_STATUS_INTERNAL_ERROR);
		BF_ASSERT(c_flattened.ndim == b_flattened.ndim, BF_STATUS_INTERNAL_ERROR);
		ndim = c->ndim;
		
		for( int d=0; d<ndim-2; ++d ) {
			BF_ASSERT(a->shape[d] == c->shape[d] || a->shape[d] == 1, BF_STATUS_INVALID_SHAPE);
			BF_ASSERT(b->shape[d] == c->shape[d] || b->shape[d] == 1, BF_STATUS_INVALID_SHAPE);
			batch_shape[d] = c->shape[d];
			// Find longest dimension to use as kernel batch dim
			if( c->shape[d] >= nbatch ) {
				nbatch = c->shape[d];
				batch_dim = d;
			}
		}
		// Remove the kernel batch dim from the rest of the batch shape
		batch_shape[batch_dim] = 1;
	}
	
	// Convert byte strides to element strides
	int astrides[BF_MAX_DIMS];
	int bstrides[BF_MAX_DIMS];
	int cstrides[BF_MAX_DIMS];
	for( int d=0; d<ndim ; ++d ) {
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
		// TODO: I think this actually occurs legitimately when shape[-1] = 1
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
	BF_ASSERT(cstrides[ndim-2] >= cstrides[ndim-1], BF_STATUS_UNSUPPORTED_STRIDE);
	
	if( nbatch > 1 ) {
		// Enable broadcasting in the kernel batch dim
		if( a->shape[batch_dim] == 1 ) { astrides[batch_dim] = 0; }
		if( b->shape[batch_dim] == 1 ) { bstrides[batch_dim] = 0; }
	}
	
	ShapeIndexer<BF_MAX_DIMS> shape_indexer(batch_shape, ndim);
	for( long i=0; i<shape_indexer.size(); ++i ) {
		auto inds = shape_indexer.at(i);
		void* a_data = array_get_pointer(a, inds);
		void* b_data = array_get_pointer(b, inds);
		void* c_data = array_get_pointer(c, inds);
		cuda::child_stream stream(g_cuda_stream);
		BF_CHECK( bfMatMul_ab_exec(handle, stream, trans_b, trans_a,
		                           c->shape[ndim-1], // m
		                           c->shape[ndim-2], // n
		                           a->shape[ndim-1], // k
		                           nbatch,
		                           alpha,
		                           // Note: We swap a and b here because
		                           //         CUBLAS uses column-major
		                           //         while we use row-major order.
		                           b_data, b->dtype, bstrides[ndim-2], bstrides[batch_dim],
		                           a_data, a->dtype, astrides[ndim-2], astrides[batch_dim],
		                           beta,
		                           c_data, c->dtype, cstrides[ndim-2], cstrides[batch_dim]) );
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
// Computes c = a.b, or a.a^H or b^H.b if either a or b are NULL
BFstatus bfLinAlgMatMul(BFlinalg       handle,
                        double         alpha,
                        BFarray const* a,   // [...,i,j]
                        BFarray const* b,   // [...,j,k]
                        double         beta,
                        BFarray const* c) {  // [...,i,k]
	BF_TRACE();
	BF_ASSERT(handle, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(a || b, BF_STATUS_INVALID_ARGUMENT);
	BF_ASSERT(c, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(space_accessible_from(c->space, BF_SPACE_CUDA),
	          BF_STATUS_UNSUPPORTED_SPACE);
	if( a && b ) {
		BF_ASSERT(space_accessible_from(a->space, BF_SPACE_CUDA),
	          BF_STATUS_UNSUPPORTED_SPACE);
		BF_ASSERT(space_accessible_from(b->space, BF_SPACE_CUDA),
		          BF_STATUS_UNSUPPORTED_SPACE);
		return bfMatMul_ab(handle, alpha, a, b, beta, c);
		//BF_TRY_RETURN(bfMatMul_ab(handle, alpha, a, b, beta, c));
	} else {
		BFarray const* input = a ? a : b;
		BF_ASSERT(space_accessible_from(input->space, BF_SPACE_CUDA),
		          BF_STATUS_UNSUPPORTED_SPACE);
		bool adjoint = (input == b);
		// TODO: BF_TRY_RETURN
		return bfMatMul_aa(handle, alpha, input, adjoint, beta, c);
	}
}
