#include <bifrost/common.h>
#include <bifrost/array.h>

BFstatus bfXgpuInitialize(BFarray *in, BFarray *out, int gpu_dev);
BFstatus bfXgpuCorrelate(BFarray *in, BFarray *out, int doDump);
BFstatus bfXgpuKernel(BFarray *in, BFarray *out, int doDump);
BFstatus bfXgpuSubSelect(BFarray *in, BFarray *out, BFarray *vismap, BFarray *conj, int nchan_sum);
BFstatus bfXgpuGetOrder(BFarray *antpol_to_input, BFarray *antpol_to_bl, BFarray *is_conj);
BFstatus bfXgpuReorder(BFarray *xgpu_output, BFarray *reordered, BFarray *baselines, BFarray *is_conjugated);
