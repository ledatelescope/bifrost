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
  TODO: Clean up by abstracting some of the common pieces into separate funcs
            E.g., 8/16-bit renormalization
                  alpha/beta application

  TODO: Try 2D tex fetch

Cmn = Amk Akn^H
m is the fast dim of C (and A)
  This means the conjugated dim is the slow dim
    cublasCherk does not support conjugating, so we're stuck with this
*/

#include "utils.hpp"
#include "Complex.hpp"
#include "Jones.hpp"
#include "assert.hpp"
#include "linalg_kernels.h"
#include "cuda/stream.hpp"

#define BF_USE_DIAGONAL_KERNEL 1
//#define BF_USE_DIAGONAL_KERNEL 0

inline __host__ __device__
int project_triangular(int i, int j) {
	// Note: Assumes i >= j
	// Note: i=slow, j=fast => lower triangle
	return i*(i+1)/2 + j;
}
inline __host__ __device__
void lift_triangular(int b, int* i, int* j) {
	// Note: Returned values obey i >= j
	// Note: i=slow, j=fast => lower triangle
	// Warning: This works up to nant=4607; beyond that, float64 is required
	*i = int((sqrtf(8*b+1)-1)/2);
	*j = b - project_triangular(*i, 0);
}

template<typename T>
__host__ __device__
inline T correct_output_for_tex_normalization(T const& output, int input_nbit) {
	T ret = output;
	if( input_nbit == 8 || input_nbit == 16 ) {
		// Correct for normalization applied during tex load
		//   Note that the (2**n)-1 scale factor used in the
		//     tex load introduces noise into the input
		//     mantissas, which results in increased loss of
		//     precision during the computation. We fix this by
		//     explicitly rounding the result here to an
		//     integer.
		int input_scale_factor  = (1 << (input_nbit - 1)) - 1;
		int output_scale_factor = (input_scale_factor *
		                           input_scale_factor);
		ret = rintf(output * float(output_scale_factor));
	}
	return ret;
}

template<int M_REG, int N_REG, int M_THREAD, int N_THREAD>
__device__
inline void bf_cherk_N_diagonal_kernel_compute(int tm,
                                               int tn,
                                               int tm0_extra,
                                               int tn0_extra,
                                               Complex<float> const& data,
                                               JonesMat<float> C[N_REG][M_REG],
                                               JonesRowVec<float>& C_extra) {
	JonesVec<float> A[M_REG];
	JonesVec<float> B[N_REG];
#pragma unroll
	for( int rm=0; rm<M_REG; ++rm ) {
		A[rm] = JonesVec<float>(__shfl(data, (tm*M_REG + rm)*2 + 0),
		                        __shfl(data, (tm*M_REG + rm)*2 + 1));
	}
#pragma unroll
	for( int rn=0; rn<N_REG; ++rn ) {
		B[rn] = JonesVec<float>(__shfl(data, (tn*N_REG + rn)*2 + 0),
		                        __shfl(data, (tn*N_REG + rn)*2 + 1));
	}
#pragma unroll
	for( int rn=0; rn<N_REG; ++rn ) {
#pragma unroll
		for( int rm=0; rm<M_REG; ++rm ) {
			C[rn][rm].mad(B[rn].conj(), A[rm].H().conj());
		}
	}
	
	// Note: Only the first 16 threads will write out C_extra
	JonesVec<float> A_extra(__shfl(data, tm0_extra*2 + 0),
	                        __shfl(data, tm0_extra*2 + 1));
	Complex<float> B_extra = __shfl(data, tn0_extra);
	C_extra.x.mad(A_extra.x, B_extra.conj());
	C_extra.y.mad(A_extra.y, B_extra.conj());
}

// This kernel works by mapping a single warp to the lower triangle of a
//   block. The 32 threads each work on a 2x2 JonesMat reg tile, which leaves
//   4 tiles uncomputed. Elements of these extra tiles are computed by the
//   first 16 threads. Unfortunately this has quite a large impact on perf.
template<int M_THREAD, int N_THREAD, int M_REG, int N_REG>
__global__
__launch_bounds__(M_THREAD*N_THREAD)
void bf_cherk_N_diagonal_kernel(int N,
                                int K,
                                int nbatch,
                                float alpha,
                                cudaTextureObject_t A_tex,
                                int A_nbit,
                                int A_stride,
                                int A_batchstride,
                                float beta,
                                float4* __restrict__ d_C,
                                int C_stride,
                                int C_batchstride) {
	int M = N;
	enum {
		NBUF   = 4, // Must be 4
	};
	int bid = blockIdx.x;
	int tid = threadIdx.x + threadIdx.y * M_THREAD;
	int bm, bn;
	int tm, tn;
	bm = bid;
	bn = bid;
	int tid_tile = tid;
	// Note: This maps threads such that the 4 leftover register tiles end up
	//         as a square in the bottom corner.
	if( tid_tile >= 21 ) {
		tid_tile += 2;
	}
	if( tid_tile >= 28 ) {
		tid_tile += 2;
	}
	
	lift_triangular(tid_tile, &tn, &tm);
	
	int tn0_extra_out = tid / 4 + 24;
	int tm0_extra_out = tid % 4;
	int tm0_extra = tm0_extra_out;
	int tn0_extra = tn0_extra_out;
	
	// Buffers for input row(=col) data, stored in regs and accessed via shfl
	Complex<float> data[NBUF];
	
	int input_offset0 = bm*(M_THREAD*M_REG)*2 + tid;
	
#define CHERK_LOAD(buf, k) \
	data[buf] = \
		tex1Dfetch<float2>(A_tex, (k)*A_stride*2 + input_offset)
	
#define CHERK_COMPUTE(buf) \
	bf_cherk_N_diagonal_kernel_compute \
		<M_REG, N_REG, M_THREAD, N_THREAD> \
		(tm, tn, tm0_extra, tn0_extra, data[buf], C, C_extra)
	
	for( int batch=blockIdx.y; batch<nbatch; batch+=gridDim.y ) {
		int input_offset = input_offset0 + batch*A_batchstride*2;
		JonesMat<float> C[N_REG][M_REG];
		JonesRowVec<float> C_extra(0, 0);
#pragma unroll
		for( int rn=0; rn<N_REG; ++rn ) {
#pragma unroll
			for( int rm=0; rm<M_REG; ++rm ) {
				C[rn][rm] = JonesMat<float>(0);
			}
		}
		CHERK_LOAD(0, 0);
		CHERK_LOAD(1, 1);
		for( int k=0; k<K-NBUF; k+=NBUF ) {
			__syncthreads();
			CHERK_COMPUTE(0);
			CHERK_COMPUTE(1);
			CHERK_LOAD(2, k+2);
			CHERK_LOAD(3, k+3);
			__syncthreads();
			CHERK_COMPUTE(2);
			CHERK_COMPUTE(3);
			CHERK_LOAD(0, k+4);
			CHERK_LOAD(1, k+5);
		}
		__syncthreads();
		CHERK_COMPUTE(0);
		CHERK_COMPUTE(1);
		CHERK_LOAD(2, K-2);
		CHERK_LOAD(3, K-1);
		__syncthreads();
		CHERK_COMPUTE(2);
		CHERK_COMPUTE(3);
#undef CHERK_COMPUTE
#undef CHERK_LOAD
		
		C_extra = correct_output_for_tex_normalization(C_extra, A_nbit);
		int m = tm0_extra_out + M_REG*M_THREAD*bm;
		int n = tn0_extra_out + N_REG*M_THREAD*bn*2;
		if( n < N*2 && m < M ) {
			float4* d_C_x = &d_C[(n + 0)*C_stride + batch*C_batchstride + m];
			if( alpha != 1 ) {
				C_extra *= alpha;
			}
			if( beta != 0 ) {
				JonesRowVec<float> C_old(*d_C_x);
				C_extra += beta * C_old;
			}
			*d_C_x = C_extra;
		}
		
#pragma unroll
		for( int rn=0; rn<N_REG; ++rn ) {
			int n = rn + tn*N_REG + N_REG*M_THREAD*bn;
#pragma unroll
			for( int rm=0; rm<M_REG; ++rm ) {
				int m = rm + tm*M_REG + M_REG*M_THREAD*bm;
				if( n < N && m < M && m <= n ) {
					float4* d_C_x = &d_C[(n*2 + 0)*C_stride + batch*C_batchstride + m];
					float4* d_C_y = &d_C[(n*2 + 1)*C_stride + batch*C_batchstride + m];
					JonesMat<float>& C_new = C[rn][rm];
					
					C_new = correct_output_for_tex_normalization(C_new, A_nbit);
					if( alpha != 1 ) {
						C_new *= alpha;
					}
					if( beta != 0 ) {
						JonesMat<float> C_old(*d_C_x, *d_C_y);
						C_new += beta * C_old;
					}
					
					if( n == m ) {
						// Only write the xx term, not the yx over the diagonal
						*(Complex<float>*)d_C_x = C_new.x.x;
					} else {
						*d_C_x = C_new.x;
					}
					*d_C_y = C_new.y;
				}
			}
		}
	} // End batch loop
}

template<int M_THREAD, int N_THREAD, int M_REG, int N_REG>
__device__
inline void bf_cherk_N_offdiagonal_kernel_compute(int tm,
                                                  int tn,
                                                  JonesVec<float> const s_A[M_REG][M_THREAD],
                                                  JonesVec<float> const s_B[N_REG][M_THREAD],
                                                  JonesMat<float> C[N_REG][M_REG]) {
	JonesVec<float> A[M_REG];
	JonesVec<float> B[N_REG];
#pragma unroll
	for( int rm=0; rm<M_REG; ++rm ) {
		A[rm] = s_A[rm][tm];
	}
#pragma unroll
	for( int rn=0; rn<N_REG; ++rn ) {
		B[rn] = s_B[rn][tn];
	}
#pragma unroll
	for( int rn=0; rn<N_REG; ++rn ) {
#pragma unroll
		for( int rm=0; rm<M_REG; ++rm ) {
			C[rn][rm].mad(B[rn].conj(), A[rm].H().conj());
		}
	}
}

// Cmn = Amk Bkn (m = fastest-changing dim)
// Cherk kernel based on xGPU
// This is designed for large k
template<int M_THREAD, int N_THREAD, int M_REG, int N_REG>
__global__
__launch_bounds__(M_THREAD*N_THREAD)
void bf_cherk_N_offdiagonal_kernel(int N,
                                   int K,
                                   int nbatch,
                                   float alpha,
                                   cudaTextureObject_t A_tex,
                                   int A_nbit,
                                   int A_stride,
                                   int A_batchstride,
                                   float beta,
                                   float4* __restrict__ d_C,
                                   int C_stride,
                                   int C_batchstride) {
	enum {
		NBUF   = 4,// Must be 4
		NARRAY = 2 // Must be 2
	};
	int bid = blockIdx.x;
	int tid = threadIdx.x + threadIdx.y * M_THREAD;
	int bm, bn;
	int tm, tn;
	lift_triangular(bid, &bn, &bm);
#if BF_USE_DIAGONAL_KERNEL
	bn += 1; // Process below the diagonal
#endif
	tm = threadIdx.x;
	tn = threadIdx.y;
	
	__shared__ JonesVec<float> smem[NBUF][NARRAY][M_REG][M_THREAD];
	
	int input_offset0;
	if( tid < M_THREAD*M_REG ) {
		input_offset0 = bm*(M_THREAD*M_REG) + tid;
	} else {
		input_offset0 = bn*(M_THREAD*N_REG) + (tid - M_THREAD*M_REG);
	}
	
	// Note: This load is not bounds-checked, but it doesn't matter when using
	//         texture loads.
#define CHERK_LOAD(buf, k) \
	if( M_REG == 4 || tid < M_THREAD*M_REG * 2 ) \
		(&smem[buf][0][0][0])[tid] = \
			tex1Dfetch<float4>(A_tex, (k)*A_stride + input_offset)
	
#define CHERK_COMPUTE(buf) \
	bf_cherk_N_offdiagonal_kernel_compute \
		<M_THREAD, N_THREAD, M_REG, N_REG> \
		(tm, tn, smem[buf][0], smem[buf][1], C)
	
	for( int batch=blockIdx.y; batch<nbatch; batch+=gridDim.y ) {
		int input_offset = input_offset0 + batch*A_batchstride;
		JonesMat<float> C[N_REG][M_REG];
#pragma unroll
		for( int rn=0; rn<N_REG; ++rn ) {
#pragma unroll
			for( int rm=0; rm<M_REG; ++rm ) {
				C[rn][rm] = JonesMat<float>(0);
			}
		}
		CHERK_LOAD(0, 0);
		CHERK_LOAD(1, 1);
		for( int k=0; k<K-NBUF; k+=NBUF ) {
			__syncthreads();
			CHERK_COMPUTE(0);
			CHERK_COMPUTE(1);
			CHERK_LOAD(2, k+2);
			CHERK_LOAD(3, k+3);
			__syncthreads();
			CHERK_COMPUTE(2);
			CHERK_COMPUTE(3);
			CHERK_LOAD(0, k+4);
			CHERK_LOAD(1, k+5);
		}
		__syncthreads();
		CHERK_COMPUTE(0);
		CHERK_COMPUTE(1);
		CHERK_LOAD(2, K-2);
		CHERK_LOAD(3, K-1);
		__syncthreads();
		CHERK_COMPUTE(2);
		CHERK_COMPUTE(3);
#undef CHERK_COMPUTE
#undef CHERK_LOAD
		
#pragma unroll
		for( int rn=0; rn<N_REG; ++rn ) {
			int n = tn + M_THREAD*(rn + N_REG*bn);
#pragma unroll
			for( int rm=0; rm<M_REG; ++rm ) {
				int m = tm + M_THREAD*(rm + M_REG*bm);
				
				int M = N;
				if( n < N && m < M
#if !BF_USE_DIAGONAL_KERNEL
				    && m <= n
#endif
				    ) {
					float4* d_C_x = &d_C[(n*2 + 0)*C_stride + batch*C_batchstride + m];
					float4* d_C_y = &d_C[(n*2 + 1)*C_stride + batch*C_batchstride + m];
					JonesMat<float>& C_new = C[rn][rm];
					
					C_new = correct_output_for_tex_normalization(C_new, A_nbit);
					if( alpha != 1 ) {
						C_new *= alpha;
					}
					if( beta != 0 ) {
						JonesMat<float> C_old(*d_C_x, *d_C_y);
						C_new += beta * C_old;
					}
#if !BF_USE_DIAGONAL_KERNEL
					if( n == m ) {
						// Only write the xx term, not the yx over the diagonal
						*(Complex<float>*)d_C_x = C_new.x.x;
					} else {
						*d_C_x = C_new.x;
					}
#else
					*d_C_x = C_new.x;
#endif
					*d_C_y = C_new.y;
				}
			}
		}
	} // End batch loop
}

void bf_cherk_N(int N, int K, int nbatch,
                float alpha,
                void const* A_ptr,
                BFdtype A_type,
                int A_stride,
                int A_batchstride,
                float beta,
                void* C_ptr,
                BFdtype C_type,
                int C_stride,
                int C_batchstride,
                cudaStream_t stream) {
	// Note: The kernel operates on 2 elements at a time and requires alignment
	BF_ASSERT_EXCEPTION(K             % 4 == 0, BF_STATUS_UNSUPPORTED_SHAPE);
	BF_ASSERT_EXCEPTION(N             % 2 == 0, BF_STATUS_UNSUPPORTED_SHAPE);
	BF_ASSERT_EXCEPTION(A_stride      % 2 == 0, BF_STATUS_UNSUPPORTED_STRIDE);
	BF_ASSERT_EXCEPTION(A_batchstride % 2 == 0, BF_STATUS_UNSUPPORTED_STRIDE);
	BF_ASSERT_EXCEPTION(C_stride      % 2 == 0, BF_STATUS_UNSUPPORTED_STRIDE);
	BF_ASSERT_EXCEPTION(C_batchstride % 2 == 0, BF_STATUS_UNSUPPORTED_STRIDE);
	// TODO: Assert supported limits on N and K based on texture constraints
	
	// Note: The kernel is 2x vectorized (i.e., operates on Jones vectors)
	N /= 2;
	A_stride /= 2;
	A_batchstride /= 2;
	C_stride /= 2;
	C_batchstride /= 2;
	
	size_t A_nelement_total = std::max(A_stride * K,
	                                   A_batchstride * nbatch);
	size_t texture_element_limit = 1 << 27;
	BF_ASSERT_EXCEPTION(A_nelement_total <= texture_element_limit,
	                    BF_STATUS_UNSUPPORTED_SHAPE);
	
	cudaChannelFormatKind channel_format;
	cudaTextureReadMode   tex_read_mode;
	switch( A_type ) {
	case BF_DTYPE_CI8: // Fall-through
	case BF_DTYPE_CI16:
		channel_format = cudaChannelFormatKindSigned;
		tex_read_mode  = cudaReadModeNormalizedFloat;
		break;
	case BF_DTYPE_CF32:
		channel_format = cudaChannelFormatKindFloat;
		tex_read_mode  = cudaReadModeElementType;
		break;
	default:
		BF_FAIL_EXCEPTION("Supported input dtype",
		                  BF_STATUS_UNSUPPORTED_DTYPE);
	}
	BF_ASSERT_EXCEPTION(C_type == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
	int A_nbit = BF_DTYPE_NBIT(A_type) / 2;
	
	// Create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = const_cast<void*>(A_ptr);
	resDesc.res.linear.desc.f = channel_format;
	resDesc.res.linear.desc.x = A_nbit;
	resDesc.res.linear.desc.y = A_nbit;
	resDesc.res.linear.desc.z = A_nbit;
	resDesc.res.linear.desc.w = A_nbit;
	resDesc.res.linear.sizeInBytes = A_nelement_total * 4 * (A_nbit / 8);
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = tex_read_mode;
	cudaTextureObject_t A_tex = 0;
	cudaCreateTextureObject(&A_tex, &resDesc, &texDesc, NULL);
	
	cuda::child_stream stream0(stream);
	cuda::child_stream stream1(stream);
	
	enum {
		M_THREAD = 8,
		N_THREAD = 8, // TODO: Don't think it's possible to support this != M_THREAD
		//M_REG    = 4,
		//N_REG    = 4//2
		//M_REG    = 3, // Best at large N
		//N_REG    = 3
		M_REG    = 2,
		N_REG    = 2 // Best at small N
	};
	dim3 block(N_THREAD, M_THREAD);
	size_t nblock_m = (N - 1) / (M_THREAD * M_REG) + 1;
#if BF_USE_DIAGONAL_KERNEL
	size_t nblock = nblock_m * (nblock_m - 1) / 2;
#else
	size_t nblock = nblock_m * (nblock_m + 1) / 2;
#endif
	dim3 grid(nblock, std::min(nbatch, 65535));
	
	// TODO: Replace with cudaLaunchKernel
	bf_cherk_N_offdiagonal_kernel<M_THREAD, N_THREAD, M_REG, N_REG>
		<<<grid, block, 0, stream0>>>
		(N, K, nbatch,
		 alpha,
		 A_tex, A_nbit, A_stride, A_batchstride,
		 beta,
		 (float4*)C_ptr, C_stride, C_batchstride);
	
	// Note: The second texture has 2x the elements because each is only
	//         a 2-vector instead of a 4-vector.
	// TODO: This condition is not currently included in the linalg.cu dispatch
	//         between this and CUBLAS, even though this can fail when CUBLAS
	//         would work. Really we want to increase this limit, perhaps using
	//         2D textures, so that it becomes very rare to run into it.
	BF_ASSERT_EXCEPTION(A_nelement_total*2 <= texture_element_limit,
	                    BF_STATUS_UNSUPPORTED_SHAPE);
	// Create texture object
	cudaResourceDesc resDesc2;
	memset(&resDesc2, 0, sizeof(resDesc2));
	resDesc2.resType = cudaResourceTypeLinear;
	resDesc2.res.linear.devPtr = const_cast<void*>(A_ptr);
	resDesc2.res.linear.desc.f = channel_format;
	resDesc2.res.linear.desc.x = A_nbit;
	resDesc2.res.linear.desc.y = A_nbit;
	resDesc2.res.linear.desc.z = 0;
	resDesc2.res.linear.desc.w = 0;
	resDesc2.res.linear.sizeInBytes = (A_nelement_total*2) * 2 * (A_nbit / 8);
	cudaTextureDesc texDesc2;
	memset(&texDesc2, 0, sizeof(texDesc2));
	texDesc2.readMode = tex_read_mode;
	cudaTextureObject_t A_tex2 = 0;
	cudaCreateTextureObject(&A_tex2, &resDesc2, &texDesc2, NULL);
	
	// TODO: Clean this up a bit
	grid.x = nblock_m;
	enum { N_THREAD_DIAG = 4 };
	block.y = N_THREAD_DIAG;
#if BF_USE_DIAGONAL_KERNEL
	bf_cherk_N_diagonal_kernel<M_THREAD, N_THREAD_DIAG, 2, 2>
		<<<grid, block, 0, stream1>>>
		(N, K, nbatch,
		 alpha,
		 A_tex2, A_nbit, A_stride, A_batchstride,
		 beta,
		 (float4*)C_ptr, C_stride, C_batchstride);
#endif // BF_USE_DIAGONAL_KERNEL
	
	cudaDestroyTextureObject(A_tex2);
	cudaDestroyTextureObject(A_tex);
}

template<int SIZE> struct shflable_type {};
template<>         struct shflable_type<4> { typedef int       type; };
template<>         struct shflable_type<8> { typedef long long type; };

template<typename T, int WIDTH=32>
inline __device__ T warp_all_sum(T x) {
	typedef typename shflable_type<sizeof(T)>::type shfl_type;
#pragma unroll
	for( int k=WIDTH>>1; k>=1; k>>=1 ) {
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 9
		x += type_pun<T>(__shfl_xor_sync(0xFFFFFFFF, type_pun<shfl_type>(x), k, WIDTH));
#else
		x += type_pun<T>(__shfl_xor(type_pun<shfl_type>(x), k, WIDTH));
#endif
	}
	return x;
}

// Btcb = Asct^T Wscb
// Cmbn = Akbm^T Bkbn
// This is designed for small N (nbeam) such that B fits in shared mem,
//   K (nstand) >= 32, and large M (ntime) * nbatch enough
//   to fill the GPU.
//   It only supports TN data ordering (i.e., k (stand) is fastest dim of both
//     inputs, m (time) is fastest dim of output.
//   It is vectorized 2x so that the input elements are loaded and processed as
//     JonesVecs.
// Note: BLOCK_X must be == warpSize (32)
// Note: BLOCK_M must be <= BLOCK_X
template<int N_MAX, int BLOCK_X, int BLOCK_Y, int BLOCK_M,
         typename AlphaType, typename AType, typename BType,
         typename BetaType, typename CType>
__global__
__launch_bounds__(BLOCK_X*BLOCK_Y)
void bf_cgemm_TN_smallM_kernel_v2(int M,
                                  int N,
                                  int K,
                                  int nbatch,
                                  AlphaType alpha,
                                  AType const* __restrict__ d_A,
                                  int A_stride,
                                  int A_batchstride,
                                  BType const* __restrict__ d_B,
                                  int B_stride,
                                  int B_batchstride,
                                  BetaType beta,
                                  CType*       __restrict__ d_C,
                                  int C_stride,
                                  int C_batchstride) {
	typedef JonesVec<float> ComputeType;
	typedef Complex<float>  SumType;
	extern __shared__ char smem[];
	BType* s_B = (BType*)smem;
	int xi = threadIdx.x; // Acts as the k dim for input and a block of the m dim for output
	int y  = threadIdx.y + blockIdx.y*blockDim.y; // Blocks along the m dim
	int zi = blockIdx.z;  // Batch dimension
	int K_blocks = (K - 1) / BLOCK_X + 1;
	int s_B_stride = K_blocks * BLOCK_X;
	
	// Grid-stride loop over batches
	for( int z=zi; z<nbatch; z+=gridDim.z ) {
	
	// Cache all N*K elements of B for this batch in shared mem
	__syncthreads();
	if( threadIdx.y == 0 ) {
#pragma unroll
		for( int n=0; n<N_MAX; ++n ) {
			if( n < N ) {
				for( int x=xi; x<K_blocks*BLOCK_X; x+=BLOCK_X ) {
					BType B;
					if( x < K ) {
						B = d_B[n*B_stride + z*B_batchstride + x];
					} else {
						B = BType(0, 0);
					}
					s_B[n*s_B_stride + x] = B;
				}
			}
		}
	}
	__syncthreads();
	SumType C[N_MAX];
	// Loop through this block of M
	for( int mi=0; mi<BLOCK_M; ++mi ) {
		int m = y*BLOCK_M + mi;
		// HACK TESTING
		//if( m >= M ) {
		//	break;
		//}
		SumType C_tmp[N_MAX];
#pragma unroll
		for( int n=0; n<N_MAX; ++n ) {
			C_tmp[n] = 0;
		}
		// Loop through blocks of K
		for( int x=xi; x<K_blocks*BLOCK_X; x+=BLOCK_X ) {
			// Load a warp of elements from A
			ComputeType A(0, 0); // Note: Set to 0 to make extra threads safe
			if( m < M && x < K ) {
				A = d_A[m*A_stride + z*A_batchstride + x];
			}
			// Loop through N
#pragma unroll
			for( int n=0; n<N_MAX; ++n ) {
				if( n < N ) {
					// Load B from the shared mem cache
					ComputeType B;
					if( x < K ) {
						B = s_B[n*s_B_stride + x];
					} else {
						//B = 0; // HACK TESTING
					}
					// Compute dot product over the warp
					//*C_tmp[n] += warp_all_sum(A * B);
					C_tmp[n] += warp_all_sum(A.x * B.x + A.y * B.y);
				}
			}
		}
#pragma unroll
		for( int n=0; n<N_MAX; ++n ) {
			if( n < N ) {
				// Give the results to thread mi
				if( mi == xi ) {
					C[n] = C_tmp[n];
				}
			}
		}
	}
	// The first BLOCK_M threads now hold the results
	if( xi < BLOCK_M ) {
		int m = y*BLOCK_M + xi;
		if( m < M ) {
#pragma unroll
			for( int n=0; n<N_MAX; ++n ) {
				if( n < N ) {
					CType* d_C_nzm = &d_C[n*C_stride + z*C_batchstride + m];
					if( beta == 0 ) {
						*d_C_nzm = alpha * C[n];
					} else {
						*d_C_nzm = alpha * C[n] + beta * (*d_C_nzm);
					}
				}
			}
		}
	}
	
	} // end z loop
}

template<int N_MAX>
void bf_cgemm_TN_smallM_staticN_v2(int M,
                                   int N,
                                   int K,
                                   int nbatch,
                                   float alpha,
                                   void const* d_A,
                                   BFdtype A_type,
                                   int A_stride,
                                   int A_batchstride,
                                   void const* d_B,
                                   BFdtype B_type,
                                   int B_stride,
                                   int B_batchstride,
                                   float beta,
                                   void* d_C,
                                   BFdtype C_type,
                                   int C_stride,
                                   int C_batchstride,
                                   cudaStream_t stream) {
	enum {
		BLOCK_X = 32, // Must be warpSize (32)
		BLOCK_Y = 16, // Can be tuned
		BLOCK_M = 8,  // Must be <= BLOCK_X (can be tuned within this bound)
	};
	BF_ASSERT_EXCEPTION(BLOCK_X == 32,      BF_STATUS_INTERNAL_ERROR);
	BF_ASSERT_EXCEPTION(BLOCK_M <= BLOCK_X, BF_STATUS_INTERNAL_ERROR);
	BF_ASSERT_EXCEPTION(K             % 2 == 0, BF_STATUS_UNSUPPORTED_SHAPE);
	BF_ASSERT_EXCEPTION(A_stride      % 2 == 0, BF_STATUS_UNSUPPORTED_STRIDE);
	BF_ASSERT_EXCEPTION(A_batchstride % 2 == 0, BF_STATUS_UNSUPPORTED_STRIDE);
	BF_ASSERT_EXCEPTION(B_stride      % 2 == 0, BF_STATUS_UNSUPPORTED_STRIDE);
	BF_ASSERT_EXCEPTION(B_batchstride % 2 == 0, BF_STATUS_UNSUPPORTED_STRIDE);
	K /= 2;
	A_stride /= 2;
	A_batchstride /= 2;
	B_stride /= 2;
	B_batchstride /= 2;
	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid(1, (M-1)/(BLOCK_Y*BLOCK_M)+1, std::min(nbatch, 65535));
	BF_ASSERT_EXCEPTION(grid.y < 65536, BF_STATUS_INTERNAL_ERROR);
	int K_blocks = (K - 1) / BLOCK_X + 1;
	int s_B_stride = K_blocks * BLOCK_X;
	size_t smem = N * s_B_stride * BF_DTYPE_NBYTE(B_type)*2;
	bool B_fits_in_shared_mem = (smem <= 48*1024);
	BF_ASSERT_EXCEPTION(B_fits_in_shared_mem, BF_STATUS_UNSUPPORTED);
	
	/* // TODO: Use cudaLaunchKernel instead of <<< >>>
	void* args[] = {
		&M, &K, &nbatch,
		&alpha,
		(void*)&d_A, &A_stride, &A_batchstride,
		(void*)&d_B, &B_stride, &B_batchstride,
		&beta,
		(void*)&d_C, &C_stride, &C_batchstride
	};
	*/
#define LAUNCH_BF_CGEMM_TN_SMALLM_KERNEL(AType, BType, CType) \
	bf_cgemm_TN_smallM_kernel_v2<N_MAX, BLOCK_X, BLOCK_Y, BLOCK_M> \
		<<<grid,block,smem,stream>>> \
		(M, N, K, nbatch, \
		 alpha, \
		 (AType*)d_A, A_stride, A_batchstride, \
		 (BType*)d_B, B_stride, B_batchstride, \
		 beta, \
		 (CType*)d_C, C_stride, C_batchstride)
	BF_ASSERT_EXCEPTION(C_type == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
	switch( A_type ) {
	case BF_DTYPE_CI4: {
		switch( B_type ) {
		case BF_DTYPE_CI16: {
			LAUNCH_BF_CGEMM_TN_SMALLM_KERNEL(
				JonesVec<FourBit>, JonesVec<int16_t>, Complex<float>);
			break;
		}
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 9
		case BF_DTYPE_CF16: {
			LAUNCH_BF_CGEMM_TN_SMALLM_KERNEL(
				JonesVec<FourBit>, JonesVec<half>, Complex<float>);
			break;
		}
#endif
		case BF_DTYPE_CF32: {
			LAUNCH_BF_CGEMM_TN_SMALLM_KERNEL(
				JonesVec<FourBit>, JonesVec<float>, Complex<float>);
			break;
		}
		default:
			BF_FAIL_EXCEPTION("Supported dtype for B",
			                  BF_STATUS_UNSUPPORTED_DTYPE);
		}
	break;
	}
	case BF_DTYPE_CI8: {
		switch( B_type ) {
		case BF_DTYPE_CI16: {
			LAUNCH_BF_CGEMM_TN_SMALLM_KERNEL(
				JonesVec<int8_t>, JonesVec<int16_t>, Complex<float>);
			break;
		}
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 9
		case BF_DTYPE_CF16: {
			LAUNCH_BF_CGEMM_TN_SMALLM_KERNEL(
				JonesVec<int8_t>, JonesVec<half>, Complex<float>);
			break;
		}
#endif
		case BF_DTYPE_CF32: {
			LAUNCH_BF_CGEMM_TN_SMALLM_KERNEL(
				JonesVec<int8_t>, JonesVec<float>, Complex<float>);
			break;
		}
		default:
			BF_FAIL_EXCEPTION("Supported dtype for B",
			                  BF_STATUS_UNSUPPORTED_DTYPE);
		}
	break;
	}
	default:
		BF_FAIL_EXCEPTION("Supported dtype for A",
		                  BF_STATUS_UNSUPPORTED_DTYPE);
	}
#undef LAUNCH_BF_CGEMM_TN_SMALLM_KERNEL
}

void bf_cgemm_TN_smallM(int M,
                        int N,
                        int K,
                        int nbatch,
                        float alpha,
                        void const* d_A,
                        BFdtype A_type,
                        int A_stride,
                        int A_batchstride,
                        void const* d_B,
                        BFdtype B_type,
                        int B_stride,
                        int B_batchstride,
                        float beta,
                        void*       d_C,
                        BFdtype C_type,
                        int C_stride,
                        int C_batchstride,
                        cudaStream_t stream) {
	#define CALL_BF_CGEMM_TN_SMALLM_STATICN(N_MAX) \
		/*bf_cgemm_TN_smallM_staticN<N_MAX>*/ \
			bf_cgemm_TN_smallM_staticN_v2<N_MAX> \
			(M, N, K, nbatch, \
			 alpha, \
			 d_A, A_type, A_stride, A_batchstride, \
			 d_B, B_type, B_stride, B_batchstride, \
			 beta, \
			 d_C, C_type, C_stride, C_batchstride, \
			 stream)
	if(      N <=  1 ) { CALL_BF_CGEMM_TN_SMALLM_STATICN( 1); }
	else if( N <=  2 ) { CALL_BF_CGEMM_TN_SMALLM_STATICN( 2); }
	else if( N <=  4 ) { CALL_BF_CGEMM_TN_SMALLM_STATICN( 4); }
	else if( N <=  8 ) { CALL_BF_CGEMM_TN_SMALLM_STATICN( 8); }
	else if( N <= 16 ) { CALL_BF_CGEMM_TN_SMALLM_STATICN(16); }
	else { BF_FAIL_EXCEPTION("Supported N in bf_cgemm_TN_smallM",
	                         BF_STATUS_UNSUPPORTED_SHAPE); }
#undef CALL_BF_CGEMM_TN_SMALLM_STATICN
}
