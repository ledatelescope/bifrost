#include <bifrost/common.h>
#include <bifrost/array.h>

BFstatus beamformInitialize(
  int gpudev,
  int ninputs,
  int nchans,
  int ntimes,
  int nbeams,
  int ntime_blocks
);

BFstatus beamformRun(
  BFarray *in,
  BFarray *out,
  BFarray *weights
);
