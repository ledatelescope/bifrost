#ifndef BF_ROMEIN_H_INCLUDE_GUARD_
#define BF_ROMEIN_H_INCLUDE_GUARD_

// Bifrost Includes
#include <bifrost/common.h>
#include <bifrost/memory.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BFromein_impl* BFromein;

BFstatus bfRomeinCreate(BFromein* plan);
BFstatus bfRomeinInit(BFromein       plan,
                      BFarray const* positions,
                      BFarray const* kernels,
                      BFsize         ngrid,
                      BFbool         polmajor);
BFstatus bfRomeinSetStream(BFromein    plan,
                           void const* stream);
BFstatus bfRomeinSetPositions(BFromein       plan, 
                              BFarray const* positions);
BFstatus bfRomeinSetKernels(BFromein       plan, 
                            BFarray const* kernels);
BFstatus bfRomeinExecute(BFromein          plan,
                         BFarray const* in,
                         BFarray const* out);
BFstatus bfRomeinDestroy(BFromein plan);

#ifdef __cplusplus
}
#endif

#endif // BF_ROMEIN_H_INCLUDE_GUARD
