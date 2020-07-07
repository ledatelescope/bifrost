#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>

#include <bifrost/array.h>
#include <bifrost/common.h>
#include <utils.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <xgpu.h>

extern "C" {

static XGPUContext context;
static XGPUInfo info;

/* 
 * Initialize the xGPU library by providing
 * a pointer to the input and output data (on the host),
 * and a GPU device ID
 */
BFstatus bfXgpuInitialize(BFarray *in, BFarray *out, int gpu_dev) {
  int xgpu_error;
  xgpuInfo(&info);
  // Don't bother checking sizes if the input space is CUDA.
  // We're not going to use these arrays anyway
  if (in->space != BF_SPACE_CUDA) {
      if (num_contiguous_elements(in) != info.vecLength) {
        fprintf(stderr, "ERROR: xgpuInitialize: number of elements in != vecLength\n");
        fprintf(stderr, "number of elements in: %lu\n", num_contiguous_elements(in));
        fprintf(stderr, "vecLength: %llu\n", info.vecLength);
        return BF_STATUS_INVALID_SHAPE;
      }
      if (num_contiguous_elements(out) != info.matLength) {
        fprintf(stderr, "ERROR: xgpuInitialize: number of elements out != matLength\n");
        fprintf(stderr, "number of elements out: %lu\n", num_contiguous_elements(out));
        fprintf(stderr, "matLength: %llu\n", info.matLength);
        return BF_STATUS_INVALID_SHAPE;
      }
  }
  context.array_h = (SwizzleInput *)in->data;
  context.array_len = info.vecLength;
  context.matrix_h = (Complex *)out->data;
  context.matrix_len = info.matLength;
  if (in->space == BF_SPACE_CUDA) {
      xgpu_error = xgpuInit(&context, gpu_dev | XGPU_DONT_REGISTER | XGPU_DONT_MALLOC_GPU);
  } else {
      xgpu_error = xgpuInit(&context, gpu_dev);
  }
  if (xgpu_error != XGPU_OK) {
    fprintf(stderr, "ERROR: xgpuInitialize: call returned %d\n", xgpu_error);
    return BF_STATUS_INTERNAL_ERROR;
  } else {
    return BF_STATUS_SUCCESS;
  }
}

/*
 * Call the xGPU kernel.
 * in : pointer to input data array on host
 * out: pointer to output data array on host
 * doDump : if 1, this is the last call in an integration, and results
 *          will be copied to the host.
 */
BFstatus bfXgpuCorrelate(BFarray *in, BFarray *out, int doDump) {
  if (in->space == BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  if (out->space == BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  int xgpu_error;
  context.array_h = (SwizzleInput *)in->data;
  context.array_len = info.vecLength;
  context.matrix_h = (Complex *)out->data;
  context.matrix_len = info.matLength;
  xgpu_error = xgpuCudaXengineSwizzle(&context, doDump ? SYNCOP_DUMP : SYNCOP_SYNC_TRANSFER);
  if (doDump) {
    xgpuClearDeviceIntegrationBuffer(&context);
  }
  if (xgpu_error != XGPU_OK) {
    return BF_STATUS_INTERNAL_ERROR;
  } else {
    return BF_STATUS_SUCCESS;
  }
}

/*
 * Call the xGPU kernel having pre-copied data to device memory.
 * Note that this means xGPU can't take advantage of its inbuild
 * copy/compute pipelining.
 * in : pointer to input data array on device
 * out: pointer to output data array on device
 * doDump : if 1, this is the last call in an integration, and results
 *          will be copied to the host.
 */
static int newAcc = 1; // flush vacc on the first call
BFstatus bfXgpuKernel(BFarray *in, BFarray *out, int doDump) {
  if (in->space != BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  if (out->space != BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  int xgpu_error;
  context.array_h = (ComplexInput *)in->data;
  context.array_len = info.vecLength;
  context.matrix_h = (Complex *)out->data;
  context.matrix_len = info.matLength;
  xgpu_error = xgpuCudaXengineSwizzleKernel(&context, doDump ? SYNCOP_DUMP : 0, newAcc,
		                           (SwizzleInput *)in->data, (Complex *)out->data);

  if (newAcc) {
    newAcc = 0;
  }
  if (doDump) {
    newAcc = 1;
  }
  if (xgpu_error != XGPU_OK) {
    fprintf(stderr, "ERROR: xgpuKernel: kernel call returned %d\n", xgpu_error);
    return BF_STATUS_INTERNAL_ERROR;
  } else {
    return BF_STATUS_SUCCESS;
  }
}

/*
 * Given an xGPU accumulation buffer, grab a subset of visibilities from
 * and gather them in a new buffer, in order chan x visibility x complexity [int32]
 * BFarray *in : Pointer to a BFarray with storage in device memory, where xGPU results reside
 * BFarray *in : Pointer to a BFarray with storage in device memory where collated visibilities should be written.
 * int **vismap : array of visibilities in [[polA, polB], [polC, polD], ... ] form.
 * int nvis : The number of visibilities to colate (length of the vismap array)
 */
BFstatus bfXgpuSubSelect(BFarray *in, BFarray *out, BFarray *vismap) {
  long long unsigned nvis = num_contiguous_elements(vismap);
  int xgpu_error;
  if (in->space != BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  if (out->space != BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  if (vismap->space != BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  xgpu_error = xgpuCudaSubSelect(&context, (Complex *)in->data, (Complex *)out->data, (int *)vismap->data, nvis);
  if (xgpu_error != XGPU_OK) {
    fprintf(stderr, "ERROR: xgpuKernel: kernel call returned %d\n", xgpu_error);
    return BF_STATUS_INTERNAL_ERROR;
  } else {
    return BF_STATUS_SUCCESS;
  }
}

} // C
