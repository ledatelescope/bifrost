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

/*! \file common.h
 *  \brief Common definitions used throughout the library
 */

#ifndef BF_COMMON_H_INCLUDE_GUARD_
#define BF_COMMON_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

//typedef int                BFstatus;
typedef int                BFbool;
//typedef int                BFenum;
typedef float              BFcomplex[2];
typedef float              BFreal;
//typedef uint64_t           BFsize; // TODO: Check this on TK1 (32 bit)
typedef unsigned long      BFsize;
//typedef size_t             BFsize;
typedef unsigned long long BFoffset;
//typedef uint64_t BFoffset;
//typedef unsigned char      BFoffset; // HACK TESTING correct offset wrapping
typedef   signed long long BFdelta;

typedef enum BFstatus_ {
	BF_STATUS_SUCCESS                                  = 0,
	BF_STATUS_END_OF_DATA                              = 1,
	BF_STATUS_INVALID_POINTER                          = 2,
	BF_STATUS_INVALID_HANDLE                           = 3,
	BF_STATUS_INVALID_ARGUMENT                         = 4,
	BF_STATUS_INVALID_STATE                            = 5,
	BF_STATUS_INVALID_SPACE                            = 6,
	BF_STATUS_INVALID_SHAPE                            = 7,
	BF_STATUS_INVALID_STRIDE                           = 8,
	BF_STATUS_INVALID_DTYPE                            = 9,
	BF_STATUS_MEM_ALLOC_FAILED                         = 16,
	BF_STATUS_MEM_OP_FAILED                            = 17,
	BF_STATUS_UNSUPPORTED                              = 18,
	BF_STATUS_UNSUPPORTED_SPACE                        = 19,
	BF_STATUS_UNSUPPORTED_SHAPE                        = 20,
	BF_STATUS_UNSUPPORTED_STRIDE                       = 21,
	BF_STATUS_UNSUPPORTED_DTYPE                        = 22,
	BF_STATUS_FAILED_TO_CONVERGE                       = 30,
	BF_STATUS_INSUFFICIENT_STORAGE                     = 31,
	BF_STATUS_DEVICE_ERROR                             = 32,
	BF_STATUS_INTERNAL_ERROR                           = 33
} BFstatus;


// Utility
const char* bfGetStatusString(BFstatus status);
BFbool      bfGetDebugEnabled();
BFstatus    bfSetDebugEnabled(BFbool enabled);
BFbool      bfGetCudaEnabled();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_COMMON_H_INCLUDE_GUARD_
