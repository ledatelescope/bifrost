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

/*! \file unpack.h
 *  \brief A function for unpacking 1/2/4-bit data to 8-bits
 */

#ifndef BF_UNPACK_H_INCLUDE_GUARD_
#define BF_UNPACK_H_INCLUDE_GUARD_

#include <bifrost/common.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \p bfUnpack unpacks 1/2/4-bit data to 8-bits
 *
 *  \param in        Input array with 1/2/4-bit datatype of kind i/u/ci
 *  \param out       Output array with corresponding 8-bit datatype
 *  \param align_msb If true, the MSB of each input value is aligned with the
 *                   MSB in the output value (i.e., the input value is placed
 *                   into the highest bits of the output, effectively scaling
 *                   it up). If false, the output will have the same values
 *                   as the input (i.e., each input value is placed into the
 *                   lowest bits of its output value). Setting align_msb=true
 *                   results in slightly faster performance.
*/
BFstatus bfUnpack(BFarray const* in,
                  BFarray const* out,
                  BFbool         align_msb);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_UNPACK_H_INCLUDE_GUARD_
