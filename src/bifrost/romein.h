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
                      BFsize         polmajor);
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

/*****************************
    Host Functions
 *****************************/

BFstatus romein_float(BFarray const* data,
		      BFarray const* uvgrid,
		      BFarray const* illum,
		      BFarray const* data_xloc,
		      BFarray const* data_yloc,
		      BFarray const* data_zloc,
		      int max_support,
		      int grid_size,
		      int data_size,
		      int nbatch);

#ifdef __cplusplus
}
#endif

#endif // BF_ROMEIN_H_INCLUDE_GUARD
