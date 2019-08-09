#include <bifrost/common.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

    BFstatus BeanFarmer(BFarray *voltages, BFarray *weights, BFarray *beamformed_out, const int NACCUMULATE);

#ifdef __cplusplus
}
#endif
