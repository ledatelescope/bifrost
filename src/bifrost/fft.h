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

#include <cufft.h>
#include <cuda.h>
#include "cuda/stream.hpp"
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#define FFT_FORWARD CUFFT_FORWARD
#define FFT_INVERSE CUFFT_INVERSE
#define FFT_C2C CUFFT_C2C 
#define FFT_R2C CUFFT_R2C 
#define FFT_C2R CUFFT_C2R 
#include <bifrost/common.h>
#include <bifrost/ring.h>

extern "C" {
BFstatus bfFFTC2C1d(
    void** input_data, void** output_data, 
    BFsize nelements, int direction);
BFstatus bfFFTC2C2d(
    void** input_data, void** output_data, 
    BFsize nelements_x, BFsize nelements_y, 
    int direction);
BFstatus bfFFTR2C1d(
    void** input_data, void** output_data, 
    BFsize nelements);
BFstatus bfFFTR2C2d(
    void** input_data, void** output_data, 
    BFsize nelements_x, BFsize nelements_y);
BFstatus bfFFT(
    BFarray *input, BFarray *output, int direction);
}
