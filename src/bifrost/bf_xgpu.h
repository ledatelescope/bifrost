#ifndef BF_XGPU_H_INCLUDE_GUARD_
#define BF_XGPU_H_INCLUDE_GUARD_

#include <bifrost/common.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

//TODO: figure out how to make ctypesgen to the right thing with python generation
//#if(BF_XGPU_ENABLED)
BFstatus bfXgpuInitialize(BFarray *in, BFarray *out, int gpu_dev);
BFstatus bfXgpuCorrelate(BFarray *in, BFarray *out, int doDump);
BFstatus bfXgpuKernel(BFarray *in, BFarray *out, int doDump);
BFstatus bfXgpuSubSelect(BFarray *in, BFarray *out, BFarray *vismap, BFarray *conj, int nchan_sum);
BFstatus bfXgpuGetOrder(BFarray *antpol_to_input, BFarray *antpol_to_bl, BFarray *is_conj);
BFstatus bfXgpuReorder(BFarray *xgpu_output, BFarray *reordered, BFarray *baselines, BFarray *is_conjugated);
//#endif // BF_XGPU_ENABLED

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_XGPU_H_INCLUDE_GUARD
