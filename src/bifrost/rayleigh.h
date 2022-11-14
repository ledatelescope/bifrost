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

#ifndef BF_RAYLEIGH_H_INCLUDE_GUARD_
#define BF_RAYLEIGH_H_INCLUDE_GUARD_

#include <bifrost/common.h>
#include <bifrost/memory.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BFrayleigh_impl* BFrayleigh;

BFstatus bfRayleighCreate(BFrayleigh* plan);
BFstatus bfRayleighInit(BFrayleigh plan,
                        BFsize     nantpol,
                        float      alpha,
                        BFsize     clip_sigmas,
                        float      max_flag_frac,
                        BFspace    space,
                        void*      plan_storage,
                        BFsize*    plan_storage_size);
BFstatus bfRayleighSetStream(BFrayleigh  plan,
                             void const* stream);
BFstatus bfRayleighResetState(BFrayleigh plan);
BFstatus bfRayleighExecute(BFrayleigh     plan,
                           BFarray const* in,
                           BFarray const* out,
                           BFsize*        flags);
BFstatus bfRayleighDestroy(BFrayleigh plan);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_RAYLEIGH_H_INCLUDE_GUARD_
