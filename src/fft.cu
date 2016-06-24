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

/*! \file fft.cu
 *  \brief This file wraps cufft functionality into the Bifrost C++ API.
 */
#include <cufft.h>
#if BF_CUDA_ENABLED
    #include "cuda/stream.hpp"
    #include <cuda_runtime_api.h>
    #define FFT_FORWARD CUFFT_FORWARD
    #define FFT_INVERSE CUFFT_INVERSE
    #define FFT_C2C CUFFT_C2C 
    #define FFT_R2C CUFFT_R2C 
    #define FFT_C2R CUFFT_C2R 
#endif
#include <bifrost/common.h>
#include <bifrost/ring.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#define BF_MAX_DIM 3

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

typedef float BFcomplex[2];
typedef float BFreal;

/*! \brief Calls a 1 dimensional CUDA FFT.
 *
 *  @param[in] input_data 
 *  \parblock
 *  Pointer to one dimensional array
 *  of untransformed data. 
 *
 *  This input_data must be signed 32 floating point.
 *  \endparblock
 *  @param[in] nelements Number of elements in input array
 *  @param[in] direction (FFT_FORWARD/FFT_INVERSE)
 *  @param[out] output_data 
 *  \parblock
 *  Pointer to one dimensional array to hold ouput.
 * 
 *  Must be pre-allocated.
 *  \endparblock
 *  \returns Whether or not computation was a success.
 */
BFstatus bfFFTC2C1d(
    void** input_data, void** output_data, 
    BFsize nelements, int direction)
{
    cufftComplex* idata = *((cufftComplex**)input_data);
    cufftComplex* odata = *((cufftComplex**)output_data);
    cufftHandle plan;
    cufftPlan1d(&plan, nelements, CUFFT_C2C, 1);
    cufftExecC2C(plan, idata, odata, direction);
    return BF_STATUS_SUCCESS;
}

/*! \brief Calls a 2 dimensional CUDA FFT.
 *
 *  @param[in] input_data 
 *  \parblock
 *  Pointer to two dimensional array
 *  of untransformed data. 
 *
 *  This input_data must be signed 32 floating point.
 *  \endparblock
 *  @param[in] nelements_x Number of elements in input 
 *  along x-dimension in input array
 *  @param[in] nelements_y Number of elements in input 
 *  along y-dimension in input array
 *  @param[in] direction (FFT_FORWARD/FFT_INVERSE)
 *  @param[out] output_data 
 *  \parblock
 *  Pointer to two dimensional array to hold ouput.
 * 
 *  Must be pre-allocated.
 *  \endparblock
 *  \returns Whether or not computation was a success.
 */
BFstatus bfFFTC2C2d(
    void** input_data, void** output_data, 
    BFsize nelements_x, BFsize nelements_y, 
    int direction)
{
    cufftComplex* idata = *((cufftComplex**)input_data);
    cufftComplex* odata = *((cufftComplex**)output_data);
    cufftHandle plan;
    cufftPlan2d(&plan, nelements_x, nelements_y, CUFFT_C2C);
    cufftExecC2C(plan, idata, odata, direction);
    return BF_STATUS_SUCCESS;
}

/*! \brief Calls a 1 dimensional CUDA FFT on real input.
 *
 *  @param[in] input_data 
 *  \parblock
 *  Pointer to one dimensional array
 *  of untransformed data. 
 *
 *  This input_data must be signed 32 floating point.
 *  \endparblock
 *  @param[in] nelements Number of elements in input array
 *  @param[out] output_data 
 *  \parblock
 *  Pointer to one dimensional array to hold ouput.
 * 
 *  Must be pre-allocated.
 *  \endparblock
 *  \returns Whether or not computation was a success.
 */
BFstatus bfFFTR2C1d(
    void** input_data, void** output_data, 
    BFsize nelements)
{
    cufftReal* idata = *((cufftReal**)input_data);
    cufftComplex* odata = *((cufftComplex**)output_data);
    cufftHandle plan;
    cufftPlan1d(&plan, nelements, CUFFT_R2C, 1);
    cufftExecR2C(plan, idata, odata);
    return cudaGetLastError();
}

/*! \brief Calls a 2 dimensional CUDA FFT on real
 *  input
 *
 *  @param[in] input_data 
 *  \parblock
 *  Pointer to two dimensional array
 *  of untransformed data. 
 *
 *  This input_data must be signed 32 floating point.
 *  \endparblock
 *  @param[in] nelements_x Number of elements in input 
 *  along x-dimension in input array
 *  @param[in] nelements_y Number of elements in input 
 *  along y-dimension in input array
 *  @param[out] output_data 
 *  \parblock
 *  Pointer to two dimensional array to hold ouput.
 * 
 *  Must be pre-allocated.
 *  \endparblock
 *  \returns Whether or not computation was a success.
 */
BFstatus bfFFTR2C2d(
    void** input_data, void** output_data, 
    BFsize nelements_x, BFsize nelements_y)
{
    cufftReal* idata = *((cufftReal**)input_data);
    cufftComplex* odata = *((cufftComplex**)output_data);
    cufftHandle plan;
    cufftPlan2d(&plan, nelements_x, nelements_y, CUFFT_R2C);
    cufftExecR2C(plan, idata, odata);
    return cudaGetLastError();
}

/*! \brief Calls a complex FFT function based on 
 *          specifications in BFarrays
 *
 *  @param[in] input - pointer to BFarray that contains 
 *  data to be transformed, with description of that
 *  data
 *  @param[in] direction (FFT_FORWARD/FFT_INVERSE)
 *  @param[out] output - pointer to BFarray which will 
 *  contain the transformed data
 *  \returns Whether or not computation was a success.
 */
BFstatus bfFFT(
    BFarray *input, BFarray *output, int direction)
{
    // TODO: Move plan here.
    // TODO: Use planMany instead of plan1d.
    // TODO: Set up BF dtype enum.
    // TODO: Make this function support type conversion
    // TODO: Enable multiple GPU support.
    cufftHandle fftPlan; 
    if (input->dtype == 0)
    {
        if (input->ndim == 1)
            return bfFFTR2C1d(
                (void**)&(input->data), (void**)&(output->data),
                input->shape[0]);
        else if (input->ndim == 2)
            return bfFFTR2C2d(
                (void**)&(input->data), (void**)&(output->data),
                input->shape[0], input->shape[1]);
    }
    if (input->ndim == 1)
        return bfFFTC2C1d(
            (void**)&(input->data), (void**)&(output->data),
            input->shape[0], direction);
    else if(input->ndim == 2)
        return bfFFTC2C2d(
            (void**)&(input->data), (void**)&(output->data),
            input->shape[0], input->shape[1], direction);
    return BF_STATUS_INTERNAL_ERROR;
}

