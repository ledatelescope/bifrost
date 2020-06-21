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
BFstatus xgpuInitialize(BFarray *in, BFarray *out, int gpu_dev) {
  int xgpu_error;
  xgpuInfo(&info);
  if (num_contiguous_elements(in) != info.vecLength) {
    return BF_STATUS_INVALID_SHAPE;
  }
  if (num_contiguous_elements(out) != info.matLength) {
    return BF_STATUS_INVALID_SHAPE;
  }
  context.array_h = (SwizzleInput *)in->data;
  context.array_len = info.vecLength;
  context.matrix_h = (Complex *)out->data;
  context.matrix_len = info.matLength;
  xgpu_error = xgpuInit(&context, gpu_dev);
  if (xgpu_error != XGPU_OK) {
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
BFstatus xgpuCorrelate(BFarray *in, BFarray *out, int doDump) {
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
BFstatus xgpuKernel(BFarray *in, BFarray *out, int doDump) {
  int xgpu_error;
  context.array_h = (ComplexInput *)in->data;
  context.array_len = info.vecLength;
  context.matrix_h = (Complex *)out->data;
  context.matrix_len = info.matLength;
  xgpu_error = xgpuCudaXengineSwizzleKernel(&context, doDump ? SYNCOP_DUMP : SYNCOP_SYNC_TRANSFER,
		                           (SwizzleInput *)in->data, (Complex *)out->data);
  if (doDump) {
    xgpuClearDeviceIntegrationBuffer(&context);
  }
  if (xgpu_error != XGPU_OK) {
    return BF_STATUS_INTERNAL_ERROR;
  } else {
    return BF_STATUS_SUCCESS;
  }
}

} // C
