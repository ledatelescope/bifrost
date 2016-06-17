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
