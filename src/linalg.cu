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
	// Convert byte strides to element strides
	int shape[BF_MAX_DIMS];
	int astrides[BF_MAX_DIMS];
	int cstrides[BF_MAX_DIMS];
	for( int d=0; d<ndim ; ++d ) {
		shape[d]    = a->shape[d];
		astrides[d] = a->strides[d];
		cstrides[d] = c->strides[d];
	}
	for( int d=0; d<ndim ; ++d ) {
		BF_ASSERT(astrides[d] % BF_DTYPE_NBYTE(a->dtype) == 0, BF_STATUS_INVALID_STRIDE);
		BF_ASSERT(cstrides[d] % BF_DTYPE_NBYTE(c->dtype) == 0, BF_STATUS_INVALID_STRIDE);
		astrides[d] /= BF_DTYPE_NBYTE(a->dtype);
		cstrides[d] /= BF_DTYPE_NBYTE(c->dtype);
	}
	// Determine transposition based on strides, and update strides and shape
	cublasOperation_t trans;
	if( astrides[ndim-1] < astrides[ndim-2] ) {
		BF_ASSERT(c->shape[ndim-1] == a->shape[ndim-2], BF_STATUS_INVALID_SHAPE);
		BF_ASSERT(c->shape[ndim-2] == a->shape[ndim-2], BF_STATUS_INVALID_SHAPE);
		trans = (BF_DTYPE_IS_COMPLEX(a->dtype) ?
		         CUBLAS_OP_C :
		         CUBLAS_OP_T);
	} else if( astrides[ndim-1] > astrides[ndim-2] ) {
		BF_ASSERT(c->shape[ndim-1] == a->shape[ndim-1], BF_STATUS_INVALID_SHAPE);
		BF_ASSERT(c->shape[ndim-2] == a->shape[ndim-1], BF_STATUS_INVALID_SHAPE);
		trans = CUBLAS_OP_N;
		std::swap(astrides[ndim-1], astrides[ndim-2]);
		std::swap(   shape[ndim-1],    shape[ndim-2]);
	} else {
		BF_ASSERT(false, BF_STATUS_INVALID_STRIDE);
	}
	ShapeIndexer<BF_MAX_DIMS> shape_indexer(shape, ndim-2);
	for( long i=0; i<shape_indexer.size(); ++i ) {
		auto inds = shape_indexer.at(i);
		void* a_data = array_get_pointer(a, inds);
		void* c_data = array_get_pointer(c, inds);
		cuda::child_stream stream(g_cuda_stream);
		BF_CHECK( bfMatMul_aa_exec(handle, stream, trans,
		                           shape[ndim-2], shape[ndim-1],
		                           alpha, a_data, a->dtype, astrides[ndim-2],
		                           beta,  c_data, c->dtype, cstrides[ndim-2]) );
	}
	return BF_STATUS_SUCCESS;
}

BFstatus bfMatMul_ab(BFlinalg       handle,
                     double         alpha,
                     BFarray const* a,
                     BFarray const* b,
                     double         beta,
                     BFarray const* c) {
	// **TODO: Implement this!
	BF_FAIL("Implemented", BF_STATUS_UNSUPPORTED);
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
