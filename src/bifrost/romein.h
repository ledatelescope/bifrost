#ifndef BF_ROMEIN_H_INCLUDE_GUARD_
#define BF_ROMEIN_H_INCLUDE_GUARD_

// Bifrost Includes
#include <bifrost/common.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif




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
