#include <bifrost/common.h>
#include <bifrost/array.h>

BFstatus bfXgpuInitialize(BFarray *in, BFarray *out, int gpu_dev);
BFstatus bfXgpuCorrelate(BFarray *in, BFarray *out, int doDump);
BFstatus bfXgpuKernel(BFarray *in, BFarray *out, int doDump);
BFstatus bfXgpuSubSelect(BFarray *in, BFarray *out, BFarray *vismap, int nchan_sum);
