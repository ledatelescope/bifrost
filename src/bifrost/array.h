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

#ifndef BF_ARRAY_H_INCLUDE_GUARD_
#define BF_ARRAY_H_INCLUDE_GUARD_

#include <bifrost/common.h>
#include <bifrost/memory.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
	BF_MAX_DIMS = 8
};
typedef enum BFdtype_ {
	BF_DTYPE_NBIT_BITS      = 0x00FF,
	BF_DTYPE_TYPE_BITS      = 0x0F00,
	
	BF_DTYPE_INT_TYPE       = 0x0000,
	BF_DTYPE_UINT_TYPE      = 0x0100, // TODO: Consider removing in favour of signed bit
	BF_DTYPE_FLOAT_TYPE     = 0x0200,
	BF_DTYPE_STRING_TYPE    = 0x0300, // TODO: Use this as fixed-length byte array of up to 255 bytes
	
	//BF_DTYPE_SIGNED_BIT     = TODO
	BF_DTYPE_COMPLEX_BIT    = 0x1000,
	
	BF_DTYPE_I1    =  1 | BF_DTYPE_INT_TYPE,
	BF_DTYPE_I2    =  2 | BF_DTYPE_INT_TYPE,
	BF_DTYPE_I4    =  4 | BF_DTYPE_INT_TYPE,
	BF_DTYPE_I8    =  8 | BF_DTYPE_INT_TYPE,
	BF_DTYPE_I16   = 16 | BF_DTYPE_INT_TYPE,
	BF_DTYPE_I32   = 32 | BF_DTYPE_INT_TYPE,
	BF_DTYPE_I64   = 64 | BF_DTYPE_INT_TYPE,
	
	BF_DTYPE_U1    =   1 | BF_DTYPE_UINT_TYPE,
	BF_DTYPE_U2    =   2 | BF_DTYPE_UINT_TYPE,
	BF_DTYPE_U4    =   4 | BF_DTYPE_UINT_TYPE,
	BF_DTYPE_U8    =   8 | BF_DTYPE_UINT_TYPE,
	BF_DTYPE_U16   =  16 | BF_DTYPE_UINT_TYPE,
	BF_DTYPE_U32   =  32 | BF_DTYPE_UINT_TYPE,
	BF_DTYPE_U64   =  64 | BF_DTYPE_UINT_TYPE,
	
	BF_DTYPE_F16   =  16 | BF_DTYPE_FLOAT_TYPE,
	BF_DTYPE_F32   =  32 | BF_DTYPE_FLOAT_TYPE,
	BF_DTYPE_F64   =  64 | BF_DTYPE_FLOAT_TYPE,
	BF_DTYPE_F128  = 128 | BF_DTYPE_FLOAT_TYPE,
	
	BF_DTYPE_CI1   =   1 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,
	BF_DTYPE_CI2   =   2 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,
	BF_DTYPE_CI4   =   4 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,
	BF_DTYPE_CI8   =   8 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,
	BF_DTYPE_CI16  =  16 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,
	BF_DTYPE_CI32  =  32 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,
	BF_DTYPE_CI64  =  64 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,
	
	BF_DTYPE_CF16  =  16 | BF_DTYPE_FLOAT_TYPE | BF_DTYPE_COMPLEX_BIT,
	BF_DTYPE_CF32  =  32 | BF_DTYPE_FLOAT_TYPE | BF_DTYPE_COMPLEX_BIT,
	BF_DTYPE_CF64  =  64 | BF_DTYPE_FLOAT_TYPE | BF_DTYPE_COMPLEX_BIT,
	BF_DTYPE_CF128 = 128 | BF_DTYPE_FLOAT_TYPE | BF_DTYPE_COMPLEX_BIT
} BFdtype;
/*
typedef struct BFdtype_info_ {
	int32_t nbit;
	int32_t type;
	int8_t  is_complex;
	int8_t  is_floating_point;
	char    name[8];
} BFdtype_info;

// TODO: Implement this
BFstatus bfTypeInfo(BFdtype dtype, BFdtype_info* info); {
	BF_ASSERT(info, BF_STATUS_INVALID_POINTER);
	info->nbit       = (dtype & BF_DTYPE_NBIT_BITS);
	info->type       = (dtype & BF_DTYPE_TYPE_BITS) >> 8; // TODO: Avoid magic number
	info->is_signed  = (dtype & BF_DTYPE_SIGNED_BIT);
	info->is_complex = (dtype & BF_DTYPE_COMPLEX_BIT);
	
	}
*/

typedef struct BFarray_ {
	void*    data;
	BFspace  space;
	BFdtype  dtype;
	int      ndim;
	long     shape[BF_MAX_DIMS];   // Elements
	long     strides[BF_MAX_DIMS]; // Bytes
	BFbool   immutable;
	//BFbool   big_endian; // TODO: Better to be 'native_endian' (or 'byteswap') instead?
	BFbool   big_endian; // TODO: Better to be 'native_endian' (or 'byteswap') instead?
	BFbool   conjugated;
	// TODO: Consider this. It could potentially be used for alpha/beta
	//         in MatMul, and also for fixed-point numerics.
	//double scale;
	// TODO: Consider this. It could be used by bfMap.
	//const char* name;
} BFarray;

// Set space, dtype, ndim, shape
// Ret data, strides
BFstatus bfArrayMalloc(BFarray* array);

BFstatus bfArrayFree(const BFarray* array);

BFstatus bfArrayCopy(const BFarray* dst,
                     const BFarray* src);

BFstatus bfArrayMemset(const BFarray* array,
                       int            value);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_ARRAY_H_INCLUDE_GUARD_
