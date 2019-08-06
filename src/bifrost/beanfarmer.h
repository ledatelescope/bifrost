#include <bifrost/common.h>
#include <bifrost/array.h>

extern "C" {

    BFstatus BeanFarmer(BFarray *voltages, BFarray *weights, BFarray *beamformed_out, const int NACCUMULATE);

}
