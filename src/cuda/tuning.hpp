/*
 * Copyright (c) 2022, The Bifrost Authors. All rights reserved.
 * Copyright (c) 2022, The University of New Mexico. All rights reserved.
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

/*
 * fdmt.cu
 */
 
 #ifndef BF_TUNING_FDMT_BLOCK_SIZE
 #define BF_TUNING_FDMT_BLOCK_SIZE 256
 #endif

 /*
  * fir.cu
  */
  
#ifndef BF_TUNING_FIR_BLOCK_SIZE
#define BF_TUNING_FIR_BLOCK_SIZE 256
#endif
  
/*
 * guantize.cu
 */
 
#ifndef BF_TUNING_GUANTIZE_BLOCK_SIZE
#define BF_TUNING_GUANTIZE_BLOCK_SIZE 512
#endif

/*
 * gunpack.cu
 */

#ifndef BF_TUNING_GUNPACK_BLOCK_SIZE
#define BF_TUNING_GUNPACK_BLOCK_SIZE 512
#endif

/*
 * linalg_kernels.cu
 */

#ifndef BF_TUNING_LINALG_CHERK_THREAD
#define BF_TUNING_LINALG_CHERK_THREAD 8
#endif

#ifndef BF_TUNING_LINALG_CHERK_M_REG
#define BF_TUNING_LINALG_CHERK_M_REG 2
#endif

#ifndef BF_TUNING_LINALG_CHERK_N_REG
#define BF_TUNING_LINALG_CHERK_N_REG 2
#endif

#ifndef BF_TUNING_LINALG_SMALLM_BLOCK_Y
#define BF_TUNING_LINALG_SMALLM_BLOCK_Y 16
#endif

// Must be less than 32
#ifndef BF_TUNING_LINALG_SMALLM_BLOCK_M
#define BF_TUNING_LINALG_SMALLM_BLOCK_M 8
#endif

/*
 * reduce.cu
 */
 
#ifndef BF_TUNING_REDUCE_VECTOR_BLOCK_SIZE
#define BF_TUNING_REDUCE_VECTOR_BLOCK_SIZE 128
#endif

#ifndef BF_TUNING_REDUCE_LOOP_BLOCK_SIZE
#define BF_TUNING_REDUCE_LOOP_BLOCK_SIZE 128
#endif

/*
 * romein.cu
 */

#ifndef BF_TUNING_ROMEIN_BLOCK_SIZE
#define BF_TUNING_ROMEIN_BLOCK_SIZE 8
#endif

/*
 * transpose.cu
 */

#ifndef BF_TUNING_TRANSPOSE_TILE_DIM
#define BF_TUNING_TRANSPOSE_TILE_DIM 32
#endif

#ifndef BF_TUNING_TRANSPOSE_WRITE_MAX_DIM
#define BF_TUNING_TRANSPOSE_WRITE_MAX_DIM 16
#endif

#ifndef BF_TUNING_TRANSPOSE_READ_MAX_DIM
#define BF_TUNING_TRANSPOSE_READ_MAX_DIM 16
#endif
