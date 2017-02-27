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
#include <bifrost/gpu_into_op.h>
#include <bifrost/array.h>
#include "cuda.hpp"
#include <bifrost/common.h>
#include <bifrost/ring.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#define BF_MAX_DIM 3


extern "C" {

    // Add function, taken from
    // https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/
    
    __global__
    void AddIntoKernel(float *x, float *y, int n)
    {
      int index = blockIdx.x * blockDim.x + threadIdx.x;
      int stride = blockDim.x * gridDim.x;
      for (int i = index; i < n; i += stride)
        x[i] = x[i] + y[i];
    }

    __global__
    void MultiplyIntoKernel(float *x, float *y, int n)
    {
      int index = blockIdx.x * blockDim.x + threadIdx.x;
      int stride = blockDim.x * gridDim.x;
      for (int i = index; i < n; i += stride)
        x[i] = x[i] * y[i];
    }

    __global__
    void SubtractIntoKernel(float *x, float *y, int n)
    {
      int index = blockIdx.x * blockDim.x + threadIdx.x;
      int stride = blockDim.x * gridDim.x;
      for (int i = index; i < n; i += stride)
        x[i] = x[i] - y[i];
    }

    __global__
    void DivideIntoKernel(float *x, float *y, int n)
    {
      int index = blockIdx.x * blockDim.x + threadIdx.x;
      int stride = blockDim.x * gridDim.x;
      for (int i = index; i < n; i += stride)
        x[i] = x[i] / y[i];
    }
    
    // grid config taken from
    // https://devblogs.nvidia.com/parallelforall/...
    // .../cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    void launch_add_into(float * x, float * y, int n) {
        int blockSize;   // The launch configurator returned block size 
        int minGridSize; // The minimum grid size needed to achieve the 
                         // maximum occupancy for a full device launch 
        int gridSize;    // The actual grid size needed, based on input size 
        
        blockSize = 256; // Set a default blocksize to quench warning
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                              AddIntoKernel, 0, 0); 
        // Round up according to array size 
        gridSize = (n + blockSize - 1) / blockSize; 

        AddIntoKernel<<< gridSize, blockSize >>>(x, y, n); 
    }

    void launch_subtract_into(float * x, float * y, int n) {
        int blockSize;   // The launch configurator returned block size 
        int minGridSize; // The minimum grid size needed to achieve the 
                         // maximum occupancy for a full device launch 
        int gridSize;    // The actual grid size needed, based on input size 
        
        blockSize = 256; // Set a default blocksize to quench warning
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                              SubtractIntoKernel, 0, 0); 
        // Round up according to array size 
        gridSize = (n + blockSize - 1) / blockSize; 

        SubtractIntoKernel<<< gridSize, blockSize >>>(x, y, n); 
    }

    void launch_multiply_into(float * x, float * y, int n) {
        int blockSize;   // The launch configurator returned block size 
        int minGridSize; // The minimum grid size needed to achieve the 
                         // maximum occupancy for a full device launch 
        int gridSize;    // The actual grid size needed, based on input size 
        
        blockSize = 256; // Set a default blocksize to quench warning
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                              MultiplyIntoKernel, 0, 0); 
        // Round up according to array size 
        gridSize = (n + blockSize - 1) / blockSize; 

        MultiplyIntoKernel<<< gridSize, blockSize >>>(x, y, n); 
    }

    void launch_divide_into(float * x, float * y, int n) {
        int blockSize;   // The launch configurator returned block size 
        int minGridSize; // The minimum grid size needed to achieve the 
                         // maximum occupancy for a full device launch 
        int gridSize;    // The actual grid size needed, based on input size 
        
        blockSize = 256; // Set a default blocksize to quench warning
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                              DivideIntoKernel, 0, 0); 
        // Round up according to array size 
        gridSize = (n + blockSize - 1) / blockSize; 

        DivideIntoKernel<<< gridSize, blockSize >>>(x, y, n); 
    }

   BFstatus gpu_add_into(void** xdata, void** ydata, BFsize veclen)
   {
       float* x = *((float**)xdata);
       float* y = *((float**)ydata);
       
       launch_add_into(x, y, veclen);
       BF_CHECK_CUDA(cudaGetLastError(), BF_STATUS_DEVICE_ERROR);
       return BF_STATUS_SUCCESS;
   }

   BFstatus gpu_subtract_into(void** xdata, void** ydata, BFsize veclen)
   {
       float* x = *((float**)xdata);
       float* y = *((float**)ydata);
       
       launch_subtract_into(x, y, veclen);
       BF_CHECK_CUDA(cudaGetLastError(), BF_STATUS_DEVICE_ERROR);
       return BF_STATUS_SUCCESS;
   }

   BFstatus gpu_multiply_into(void** xdata, void** ydata, BFsize veclen)
   {
       float* x = *((float**)xdata);
       float* y = *((float**)ydata);
       
       launch_multiply_into(x, y, veclen);
       BF_CHECK_CUDA(cudaGetLastError(), BF_STATUS_DEVICE_ERROR);
       return BF_STATUS_SUCCESS;
   }

   BFstatus gpu_divide_into(void** xdata, void** ydata, BFsize veclen)
   {
       float* x = *((float**)xdata);
       float* y = *((float**)ydata);
       
       launch_divide_into(x, y, veclen);
       BF_CHECK_CUDA(cudaGetLastError(), BF_STATUS_DEVICE_ERROR);
       return BF_STATUS_SUCCESS;
   }

   BFstatus bfAddInto(
       BFarray *xdata, BFarray *ydata)
   {
       int nelements = 1;
       for(int i=0; i<xdata->ndim; i++){
           nelements *= xdata->shape[i];
       }
       
    return gpu_add_into(
        (void**)&(xdata->data), (void**)&(ydata->data), nelements);
    }

    BFstatus bfSubtractInto(
        BFarray *xdata, BFarray *ydata)
    {
        int nelements = 1;
        for(int i=0; i<xdata->ndim; i++){
            nelements *= xdata->shape[i];
        }
       
     return gpu_subtract_into(
         (void**)&(xdata->data), (void**)&(ydata->data), nelements);
     }

     BFstatus bfMultiplyInto(
         BFarray *xdata, BFarray *ydata)
     {
         int nelements = 1;
         for(int i=0; i<xdata->ndim; i++){
             nelements *= xdata->shape[i];
         }
      
      return gpu_multiply_into(
          (void**)&(xdata->data), (void**)&(ydata->data), nelements);
      }

     BFstatus bfDivideInto(
         BFarray *xdata, BFarray *ydata)
     {
         int nelements = 1;
         for(int i=0; i<xdata->ndim; i++){
             nelements *= xdata->shape[i];
         }
       
      return gpu_divide_into(
          (void**)&(xdata->data), (void**)&(ydata->data), nelements);
      }

      BFstatus bfSetToZero(
          BFarray *data)
      {
          int nelements = 1;
          for(int i=0; i<data->ndim; i++){
              nelements *= data->shape[i];
          }
          cudaMemset(data->data, 0, nelements*sizeof(float));
          BF_CHECK_CUDA(cudaGetLastError(), BF_STATUS_DEVICE_ERROR);
          return BF_STATUS_SUCCESS;
       }

}
