#ifndef BF_ORVILLE_H_INCLUDE_GUARD_
#define BF_ORVILLE_INCLUDE_GUARD_

// Bifrost Includes
#include <bifrost/common.h>
#include <bifrost/memory.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BForville_impl* BForville;

BFstatus bfOrvilleCreate(BForville* plan);
BFstatus bfOrvilleInit(BForville       plan,
                       BFarray const* positions,
                       BFarray const* weights,
                       BFarray const* kernel1D,
                       BFsize         gridsize,
                       double         gridres,
                       double         gridwres,
                       BFsize         oversample,
                       BFbool         polmajor,
                       BFspace        space,
                       void*          plan_storage,
                       BFsize*        plan_storage_size);
BFstatus bfOrvilleSetStream(BForville  plan,
                           void const* stream);
BFstatus bfOrvilleGetStream(BForville plan,
                            void*     stream);
BFstatus bfOrvilleSetPositions(BForville     plan, 
                              BFarray const* positions);
BFstatus bfOrvilleGetProjectionSetup(BForville      plan,
                                     int*           ntimechan,
                                     int*           gridsize,
                                     int*           npol,
                                     int*           nplane,
                                     BFarray const* midpoints);
BFstatus bfOrvilleSetWeights(BForville      plan, 
                             BFarray const* weights);
BFstatus bfOrvilleSetKernel(BForville      plan, 
                            BFarray const* kernel1D);
BFstatus bfOrvilleExecute(BForville      plan,
                          BFarray const* in,
                          BFarray const* out);
BFstatus bfOrvilleDestroy(BForville plan);

#ifdef __cplusplus
}
#endif

#endif // BF_ORVILLE_H_INCLUDE_GUARD
