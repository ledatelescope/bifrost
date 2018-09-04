#ifndef BF_FFTSHIFT_H_INCLUDE_GUARD_
#define BF_FFTSHIFT_H_INCLUDE_GUARD_

// Bifrost Includes
#include <bifrost/common.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif



/*****************************
    Host Functions
 *****************************/

BFstatus fft_shift_2d(BFarray const *grid, int size, int batch_no);
    
#ifdef __cplusplus
}
#endif

#endif // BF_ROMEIN_H_INCLUDE_GUARD
