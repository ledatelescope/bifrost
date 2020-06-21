#include <bifrost/common.h>
#include <bifrost/array.h>

//extern "C" {

BFstatus xgpuInitialize(BFarray *in, BFarray *out, int gpu_dev);
BFstatus xgpuCorrelate(BFarray *in, BFarray *out, int doDump);
BFstatus xgpuKernel(BFarray *in, BFarray *out, int doDump);

//}
