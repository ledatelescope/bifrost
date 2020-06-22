#include <bifrost/common.h>
#include <bifrost/array.h>

BFstatus xgpuInitialize(BFarray *in, BFarray *out, int gpu_dev);
BFstatus xgpuCorrelate(BFarray *in, BFarray *out, int doDump);
BFstatus xgpuKernel(BFarray *in, BFarray *out, int doDump);
BFstatus xgpuSubSelect(BFarray *in, BFarray *out, BFarray *vismap);
