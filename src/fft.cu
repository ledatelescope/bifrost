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

/*! \file fft.cu
 *  \brief This file wraps cufft functionality into the Bifrost C++ API.
 */

#include <cufft.h>
#if BF_CUDA_ENABLED
    #include "cuda/stream.hpp"
    #include <cuda_runtime_api.h>
#endif
#include <bifrost/common.h>
#include <bifrost/ring.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "assert.hpp"
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
    BFspace space)
{
    cufftComplex* idata = *((cufftComplex**)input_data);
    cufftComplex* odata = *((cufftComplex**)output_data);
    cufftHandle plan;
    cufftPlan1d(&plan, nelements, CUFFT_C2C, 1);
    cufftExecC2C(plan, idata, odata, CUFFT_FORWARD);
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
 *  stride - number of bytes for each element
 *  space - where data is located
 *  outputs:
 *  output_data - pointer to one dimensional array 
 *       of transformed data
 *  Returns whether or not the operation was a success.
 */
BFstatus bfFFTC2C2d(
    void** input_data, void** output_data, 
    BFsize nelements_x, BFsize nelements_y, 
    unsigned dtype, BFspace space)
{
    cufftComplex* idata = *((cufftComplex**)input_data);
    cufftComplex* odata = *((cufftComplex**)output_data);
    cufftHandle plan;
    cufftPlan2d(&plan, nelements_x, nelements_y, CUFFT_C2C);
    cufftExecC2C(plan, idata, odata, CUFFT_FORWARD);
    return BF_STATUS_SUCCESS;
}

/*! \brief Calls a complex FFT function based on 
 *          specifications in BFarray
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
    BFarray *input, BFarray *output)
{
    if (input->ndim == 1)
        return bfFFTC2C1d(
            (void**)&(input->data), (void**)&(output->data),
            input->shape[0], input->dtype,
            input->space);
    else if(input->ndim == 2)
        return bfFFTC2C2d(
            (void**)&(input->data), (void**)&(output->data),
            input->shape[0], input->shape[1], input->dtype,
            input->space);
    return BF_STATUS_INTERNAL_ERROR;
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
    my_data.ndim = 2;
    my_data.strides[0] = sizeof(BFcomplex);
    my_data.strides[1] = 3*sizeof(BFcomplex);
    if (bfFFT(&my_data, &my_data) != BF_STATUS_SUCCESS)
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
    BFcomplex set_data[5] = {{5,1},{30,0},{100,0},{30,0},{0,0}};
    BFcomplex* some_data;
    cudaMalloc((void**)&some_data, sizeof(BFcomplex)*5);
    cudaMemcpy(some_data, set_data, sizeof(BFcomplex)*5, cudaMemcpyHostToDevice);
    my_data.data = some_data;
    my_data.space = BF_SPACE_CUDA;
    my_data.shape[0] = 5;
    my_data.ndim = 1;
    my_data.strides[0] = sizeof(BFcomplex);
    bfFFT(&my_data, &my_data);
    cufftComplex localdata[5]={};
    cudaMemcpy(localdata, (cufftComplex*)my_data.data, sizeof(cufftComplex)*5, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 5; i++)
        printf("%f\n",cuCrealf(localdata[i]));
    //print successfully fft'd data.
}


int main()
{
    printf("Running...\n");
    //should make a call to bffft, and print results before and after.
    //test_bffft_1d();
    test_bffft_2d();
    printf("Done\n");
    return 0;
}
