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

#ifndef BF_FFT_H_INCLUDE_GUARD_
#define BF_FFT_H_INCLUDE_GUARD_

#include <bifrost/common.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BFfft_impl* BFfft;

BFstatus bfFftCreate(BFfft* plan_ptr);
BFstatus bfFftInit(BFfft          plan,
                   BFarray const* iarray,
                   BFarray const* oarray,
                   int            ndim,
                   int     const* axes,
                   BFbool         apply_fftshift,
                   size_t*        tmp_storage_size);
// in, out = complex, complex => [i]fft
// in, out = real, complex    => rfft
// in, out = complex, real    => irfft
// in, out = real, real       => ERROR
// tmp_storage_size If NULL, library will allocate storage automatically
BFstatus bfFftExecute(BFfft          plan,
                      BFarray const* iarray,
                      BFarray const* oarray,
                      BFbool         inverse,
                      void*          tmp_storage,
                      size_t         tmp_storage_size);
BFstatus bfFftDestroy(BFfft plan);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_FFT_H_INCLUDE_GUARD_
