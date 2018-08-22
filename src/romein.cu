/*
 * Copyright (c) 2018, The Bifrost Authors. All rights reserved.
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

/* 

Implements the Romein convolutional algorithm onto a GPU using CUDA. 

*/
#include <iostream>
#include <bifrost/romein.h>
#include "romein_kernels.cuh"

#include "assert.hpp"
#include "trace.hpp"
#include "utils.hpp"
#include "cuda.hpp"
#include "cuda/stream.hpp"


// Fix this to use templates properly.
BFstatus romein_float(BFarray const* data, // Our data, strided by d
		      BFarray const* uvgrid, // Our UV Grid to Convolve onto, strided by g
		      BFarray const* illum, // Our convolution kernel.
		      BFarray const* data_xloc,
		      BFarray const* data_yloc,
		      BFarray const* data_zloc,
		      int max_support,
		      int grid_size,
		      int data_size,
		      int nbatch){


    //TODO: I think remove these as the overhead is probably quite high...
    BF_TRACE();
    BF_ASSERT(uvgrid && data && illum && data_xloc && data_yloc && data_zloc,
    	      BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(uvgrid->dtype == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(data->dtype == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(illum->dtype == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(data_xloc->dtype == BF_DTYPE_I32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(data_yloc->dtype == BF_DTYPE_I32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(data_zloc->dtype == BF_DTYPE_I32, BF_STATUS_UNSUPPORTED_DTYPE);


    BF_ASSERT(space_accessible_from(data->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(uvgrid->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(illum->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(data_xloc->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(data_yloc->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(data_zloc->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
    
    
    void const* dptr = data->data;
    void const* uvgridptr = uvgrid->data;
    void const* illumptr = illum->data;
    void const* xloc = data_xloc->data;
    void const* yloc = data_yloc->data;
    void const* zloc = data_zloc->data;
    cuda::child_stream stream(g_cuda_stream);
    BF_TRACE_STREAM(stream);

    scatter_grid_kernel <<< nbatch, 16, 0, stream >>> ((cuComplex*)dptr,
						  (cuComplex*)uvgridptr,
						  (cuComplex*)illumptr,
						  (int*)xloc,
						  (int*)yloc,
						  (int*)zloc,
						  max_support,
						  grid_size,
						  data_size);
    cudaDeviceSynchronize();
    //cudaError_t err = cudaGetLastError();
    //std::cout << "Error: " << cudaGetErrorString(err) << "\n";
    return BF_STATUS_SUCCESS;
}
		      
