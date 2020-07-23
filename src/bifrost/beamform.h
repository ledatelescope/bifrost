#include <bifrost/common.h>
#include <bifrost/array.h>

BFstatus bfBeamformInitialize(
  int gpudev,
  int ninputs,
  int nchans,
  int ntimes,
  int nbeams,
  int ntime_blocks
);

BFstatus bfBeamformRun(
  BFarray *in,
  BFarray *out,
  BFarray *weights
);

BFstatus bfBeamformIntegrate(
  BFarray *in,
  BFarray *out
);
