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

/*! \file common.h
 *  \brief Common definitions used throughout the library
 */

#ifndef BF_COMMON_H_INCLUDE_GUARD_
#define BF_COMMON_H_INCLUDE_GUARD_
#define BF_MAX_DIM 3

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int                BFstatus;
typedef int                BFbool;
typedef int                BFenum;
typedef float              BFcomplex[2];
typedef float              BFreal;
typedef uint64_t           BFsize; // TODO: Check this on TK1 (32 bit)
//typedef unsigned long      BFsize;
//typedef size_t             BFsize;
//typedef unsigned long long BFoffset;
typedef uint64_t BFoffset;
//typedef unsigned char      BFoffset; // HACK TESTING correct offset wrapping
typedef   signed long long BFdelta;
enum {
	BF_SPACE_AUTO         = 0,
	BF_SPACE_SYSTEM       = 1, // aligned_alloc
	BF_SPACE_CUDA         = 2, // cudaMalloc
	BF_SPACE_CUDA_HOST    = 3, // cudaHostAlloc
	BF_SPACE_CUDA_MANAGED = 4  // cudaMallocManaged
};

typedef BFenum BFspace;
/// Defines a single atom of data to be passed to a function.
typedef struct BFarray_ {
    /*! The data pointer can point towards any type of data, 
     *  so long as there is a corresponding definition in dtype. 
     *  This data should be an ndim array, which every element of
     *  type dtype.
     */
    void* data;
    /*! Where this data is located in memory.
     *  Used to ensure that operations called are localized within
     *  that space, such as a CUDA funciton operating on device
     *  memory.
     */
    BFspace space;
    unsigned dtype;
    int ndim;
    BFsize shape[BF_MAX_DIM];
    BFsize strides[BF_MAX_DIM];
} BFarray;

enum {
	BF_STATUS_SUCCESS            = 0,
	BF_STATUS_END_OF_DATA        = 1,
	BF_STATUS_INVALID_POINTER    = 2,
	BF_STATUS_INVALID_HANDLE     = 3,
	BF_STATUS_INVALID_ARGUMENT   = 4,
	BF_STATUS_INVALID_STATE      = 5,
	BF_STATUS_MEM_ALLOC_FAILED   = 6,
	BF_STATUS_MEM_OP_FAILED      = 7,
	BF_STATUS_UNSUPPORTED        = 8,
	BF_STATUS_FAILED_TO_CONVERGE = 9,
	BF_STATUS_INTERNAL_ERROR     = 10
};

// Utility
const char* bfGetStatusString(BFstatus status);
BFbool      bfGetDebugEnabled();
BFbool      bfGetCudaEnabled();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_COMMON_H_INCLUDE_GUARD_
