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
#if BF_CUDA_ENABLED
    #include "cuda/stream.hpp"
    #include <cuda_runtime_api.h>
#endif
//#include <bifrost/affinity.h>
#include <bifrost/common.h>
#include <bifrost/ring.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "assert.hpp"

#define BATCH 10
#define NX 256
#define RANK 1

typedef struct BFarray_ {
    void* data;
    BFspace space;
    unsigned dtype;
    int ndim;
    BFsize shape[3];
    BFsize strides[3];
} BFarray;

typedef float BFcomplex[2];

BFstatus bfFFTC2C1d(
    void** input_data, void** output_data, 
    BFsize nelements, unsigned dtype,
    BFsize stride, BFspace space)
{
    cufftComplex* idata = *((cufftComplex**)input_data);
    cufftComplex* odata = *((cufftComplex**)output_data);
    cufftHandle plan;
    cufftPlan1d(&plan, nelements, CUFFT_C2C, 1);
    cufftExecC2C(plan, idata, odata, CUFFT_FORWARD);
    return BF_STATUS_SUCCESS;
}

BFstatus bfFFT(
    BFarray *input, BFarray *output)
{
    if (input->ndim == 1)
    {
        return bfFFTC2C1d(
            (void**)&(input->data), (void**)&(output->data),
            input->shape[0], input->dtype,
            input->strides[0], input->space);
    }
    return BF_STATUS_INTERNAL_ERROR;
}

void test_bffft()
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
    test_bffft();
    printf("Done\n");
    return 0;
}
