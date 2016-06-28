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
