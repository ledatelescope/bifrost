#include <bifrost/common.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

BFstatus AddStuff(BFarray *xdata, BFarray *ydata, BFarray *zdata);
BFstatus SubtractStuff(BFarray *xdata, BFarray *ydata, BFarray *zdata);

#ifdef __cplusplus
} // extern "C"
#endif
