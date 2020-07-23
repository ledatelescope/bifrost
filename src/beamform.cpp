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

#include "cublas_beamform.cuh"

extern "C" {


/* 
 * Initialize the beamformer library
 */

BFstatus bfBeamformInitialize(
  int gpudev,
  int ninputs,
  int nchans,
  int ntimes,
  int nbeams,
  int ntime_blocks
) {
  // TODO: array size checking
  // TODO: use complex data types
  cublas_beamform_init(
    gpudev,
    ninputs,
    nchans,
    ntimes,
    nbeams,
    ntime_blocks
  );
  return BF_STATUS_SUCCESS;
}

BFstatus bfBeamformRun(BFarray *in, BFarray *out, BFarray *weights) {
  if (in->space != BF_SPACE_CUDA) {
    fprintf(stderr, "Beamformer input buffer must be in CUDA space\n");
    return BF_STATUS_INVALID_SPACE;
  }
  if (out->space != BF_SPACE_CUDA) {
    fprintf(stderr, "Beamformer output buffer must be in CUDA space\n");
    return BF_STATUS_INVALID_SPACE;
  }
  if (weights->space != BF_SPACE_CUDA) {
    fprintf(stderr, "Beamformer weights buffer must be in CUDA space\n");
    return BF_STATUS_INVALID_SPACE;
  }
  cublas_beamform((unsigned char *)in->data, (float *)out->data, (float *)weights->data);
  return BF_STATUS_SUCCESS;
}
  
BFstatus bfBeamformIntegrate(BFarray *in, BFarray *out) {
  if (in->space != BF_SPACE_CUDA) {
    fprintf(stderr, "Beamformer input buffer must be in CUDA space\n");
    return BF_STATUS_INVALID_SPACE;
  }
  if (out->space != BF_SPACE_CUDA) {
    fprintf(stderr, "Beamformer output buffer must be in CUDA space\n");
    return BF_STATUS_INVALID_SPACE;
  }
  cublas_beamform_integrate((float *)in->data, (float *)out->data);
  return BF_STATUS_SUCCESS;
}
} // C
