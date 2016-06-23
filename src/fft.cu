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
 *  inputs:
 *  input_data - a pointer to one dimensional array
 *       of untransformed data
 *  nelements - number of elements in in input array
 *  dtype - datatype of input array. Assumed complex.
 *  space - where data is located
 *  outputs:
 *  output_data - pointer to one dimensional array 
 *       of transformed data
 *  Returns whether or not the operation was a success.
 */
BFstatus bfFFTC2C1d(
    void** input_data, void** output_data, 
    BFsize nelements, unsigned dtype,
    BFspace space, int direction)
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
 *  inputs:
 *  input_data - a pointer to two dimensional array
 *       of untransformed data
 *  nelements_x - number of elements in input array
 *       along x dimension
 *  nelements_y - number of elements in input array
 *       along y dimension
 *  dtype - datatype of input array. Assumed complex.
 *  space - where data is located
 *  outputs:
 *  output_data - pointer to one dimensional array 
 *       of transformed data
 *  Returns whether or not the operation was a success.
 */
BFstatus bfFFTC2C2d(
    void** input_data, void** output_data, 
    BFsize nelements_x, BFsize nelements_y, 
    unsigned dtype, BFspace space, int direction)
{
    cufftComplex* idata = *((cufftComplex**)input_data);
    cufftComplex* odata = *((cufftComplex**)output_data);
    cufftHandle plan;
    cufftPlan2d(&plan, nelements_x, nelements_y, CUFFT_C2C);
    cufftExecC2C(plan, idata, odata, direction);
    return BF_STATUS_SUCCESS;
}

/*! \brief Calls a 1 dimensional real-real CUDA FFT
 *
 *  inputs:
 *  input_data - a pointer to two dimensional array
 *       of untransformed data
 *  nelements - number of elements in input array
 *  dtype - datatype of input array. Assumed complex.
 *  space - where data is located
 *  outputs:
 *  output_data - pointer to one dimensional array 
 *       of transformed data
 *  Returns whether or not the operation was a success.
 */
BFstatus bfFFTR2C1d(
    void** input_data, void** output_data, 
    BFsize nelements, unsigned dtype, 
    BFspace space)
{
    cufftReal* idata = *((cufftReal**)input_data);
    cufftComplex* odata = *((cufftComplex**)output_data);
    cufftHandle plan;
    cufftPlan1d(&plan, nelements, CUFFT_R2C, 1);
    cufftExecR2C(plan, idata, odata);
    return cudaGetLastError();
}

/*! \brief Calls a 2 dimensional real-real CUDA FFT
 *
 *  inputs:
 *  input_data - a pointer to two dimensional array
 *       of untransformed data
 *  nelements_x - number of elements in input array
 *       along x dimension
 *  nelements_y - number of elements in input array
 *       along y dimension
 *  dtype - datatype of input array. Assumed complex.
 *  stride - number of bytes for each element
 *  space - where data is located
 *  outputs:
 *  output_data - pointer to one dimensional array 
 *       of transformed data
 *  Returns whether or not the operation was a success.
 */
BFstatus bfFFTR2C2d(
    void** input_data, void** output_data, 
    BFsize nelements_x, BFsize nelements_y,
    unsigned dtype, BFspace space)
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
 *  inputs:
 *  input - pointer to BFarray that contains data to be
 *       transformed
 *  outputs:
 *  output - pointer to BFarray which will contain the
 *       transformed data
 *  Returns whether or not the operation was a success.
 */
BFstatus bfFFT(
    BFarray *input, BFarray *output, int direction)
{
    // TODO: Move plan here.
    // TODO: Make user pass FFT_R2C
    // TODO: Provide same functionality as in cufft_nyquist_packed.cu
    // TODO: Set Ben's callbacks.
    // TODO: Use planMany instead of plan1d.
    // TODO: Set up BF dtype variable.
    if (input->dtype == 0)
    {
        if (input->ndim == 1)
            return bfFFTR2C1d(
                (void**)&(input->data), (void**)&(output->data),
                input->shape[0], input->dtype,
                input->space);
        else if (input->ndim == 2)
            return bfFFTR2C2d(
                (void**)&(input->data), (void**)&(output->data),
                input->shape[0], input->shape[1],
                input->dtype, input->space);
    }
    if (input->ndim == 1)
        return bfFFTC2C1d(
            (void**)&(input->data), (void**)&(output->data),
            input->shape[0], input->dtype,
            input->space, direction);
    else if(input->ndim == 2)
        return bfFFTC2C2d(
            (void**)&(input->data), (void**)&(output->data),
            input->shape[0], input->shape[1], input->dtype,
            input->space, direction);
    return BF_STATUS_INTERNAL_ERROR;
}

void test_bffft_real_2d()
{
    BFarray my_data;
    BFarray out_data;
    BFreal set_data[3][2] = 
        {{1,2},{2,3},{3,4}};
    BFreal** some_data;
    BFcomplex* odata;
    cudaMalloc((void**)&some_data, sizeof(BFreal)*6);
    cudaMalloc((void**)&odata, sizeof(BFcomplex)*6);
    cudaMemcpy(
        some_data, set_data, 
        sizeof(BFreal)*6, cudaMemcpyHostToDevice);
    my_data.data = some_data;
    my_data.space = BF_SPACE_CUDA;
    my_data.shape[0] = 3;
    my_data.shape[1] = 2;
    my_data.dtype = 0;
    my_data.ndim = 2;
    my_data.strides[0] = 2*sizeof(BFreal);
    my_data.strides[1] = sizeof(BFreal);
    out_data = my_data;
    out_data.data = odata;
    out_data.dtype = 1;
    out_data.strides[0] = 2*sizeof(BFcomplex);
    out_data.strides[1] = sizeof(BFcomplex);
    if (bfFFT(&my_data, &out_data, FFT_FORWARD) != BF_STATUS_SUCCESS)
    {
        printf("bfFFT failed!\n");
        return; 
    }
    cufftComplex localdata[3][2] = {};
    cudaMemcpy(
        localdata, (cufftComplex*)out_data.data, 
        sizeof(cufftComplex)*6, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            printf(
                "%f+I%f\n",
                cuCrealf(localdata[i][j]),
                cuCimagf(localdata[i][j]));
        }
    }
    return;
}
void test_bffft_real()
{
    BFarray my_data;
    BFarray out_data;
    BFreal set_data[4] = {1,3,6,2.5134};
    BFreal* some_data;
    BFcomplex* odata;
    cudaMalloc((void**)&some_data, sizeof(BFreal)*5);
    cudaMalloc((void**)&odata, sizeof(BFcomplex)*3);
    cudaMemcpy(
        some_data, set_data, 
        sizeof(BFreal)*4, cudaMemcpyHostToDevice);
    my_data.data = some_data;
    my_data.space = BF_SPACE_CUDA;
    my_data.shape[0] = 4;
    my_data.dtype = 0;
    my_data.ndim = 1;
    my_data.strides[0] = sizeof(BFreal);
    out_data = my_data;
    out_data.data = odata;
    out_data.dtype = 1;
    out_data.strides[0] = sizeof(BFcomplex);
    if (bfFFT(&my_data, &out_data, FFT_FORWARD) != BF_STATUS_SUCCESS)
    {
        printf("bfFFT failed!\n");
        return; 
    }
    cufftComplex localdata[3] = {};
    cudaMemcpy(
        localdata, (cufftComplex*)out_data.data, 
        sizeof(cufftComplex)*3, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 3; i++)
        printf("%f+I%f\n",cuCrealf(localdata[i]),cuCimagf(localdata[i]));
    return;
}

void test_bffft_2d()
{
    BFarray my_data;
    BFcomplex set_data[3][3] = 
        {{{5,1},{0,0},{100,0}},
        {{5,1},{30,0},{100,0}},
        {{30,0},{0,0},{10,1}}};
    BFcomplex** some_data;
    cudaMalloc((void**)&some_data, sizeof(BFcomplex)*9);
    cudaMemcpy(
        some_data, set_data, 
        sizeof(BFcomplex)*9, cudaMemcpyHostToDevice);
    my_data.data = some_data;
    my_data.space = BF_SPACE_CUDA;
    my_data.shape[0] = 3;
    my_data.shape[1] = 3;
    my_data.dtype = 1;
    my_data.ndim = 2;
    my_data.strides[0] = 3*sizeof(BFcomplex);
    my_data.strides[1] = sizeof(BFcomplex);
    if (bfFFT(&my_data, &my_data, FFT_FORWARD) != BF_STATUS_SUCCESS)
    {
        printf("bfFFT failed!\n");
        return; 
    }
    cufftComplex localdata[3][3]={};
    cudaMemcpy(
        localdata, (cufftComplex**)my_data.data, 
        sizeof(cufftComplex)*9, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
                printf("%f\n",cuCrealf(localdata[i][j]));
    }
    //print successfully fft'd data.
    return;
}

void test_bffft_1d()
{
    BFarray my_data;
    BFcomplex set_data[5] = {{0,0},{30,0},{100,0},{30,0},{-5,0}};
    BFcomplex* some_data;
    cudaMalloc((void**)&some_data, sizeof(BFcomplex)*5);
    cudaMemcpy(some_data, set_data, sizeof(BFcomplex)*5, cudaMemcpyHostToDevice);
    my_data.data = some_data;
    my_data.space = BF_SPACE_CUDA;
    my_data.shape[0] = 5;
    my_data.dtype = 1;
    my_data.ndim = 1;
    my_data.strides[0] = sizeof(BFcomplex);
    bfFFT(&my_data, &my_data, FFT_FORWARD);
    cufftComplex localdata[5]={};
    cudaMemcpy(localdata, (cufftComplex*)my_data.data, sizeof(cufftComplex)*5, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 5; i++)
        printf("%f+I%f\n",cuCrealf(localdata[i]),cuCimagf(localdata[i]));
    //print successfully fft'd data.
}


int main()
{
    printf("Running...\n");
    //test_bffft_1d();
    //test_bffft_2d();
    //test_bffft_real();
    test_bffft_real_2d();
    printf("Done\n");
    return 0;
}
