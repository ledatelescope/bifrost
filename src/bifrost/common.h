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

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int                BFstatus;
typedef int                BFbool;
typedef int                BFenum;
typedef uint64_t BFsize; // TODO: Check this on TK1 (32 bit)
//typedef unsigned long      BFsize;
//typedef size_t             BFsize;
//typedef unsigned long long BFoffset;
typedef uint64_t BFoffset;
//typedef unsigned char      BFoffset; // HACK TESTING correct offset wrapping
typedef   signed long long BFdelta;

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
