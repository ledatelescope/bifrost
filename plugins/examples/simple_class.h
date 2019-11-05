#include <bifrost/common.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct simpleclass_impl* simpleclass;

BFstatus SimpleClassCreate(simpleclass* plan);
BFstatus SimpleClassInit(simpleclass plan,
                         float   value);
BFstatus SimpleClassExecute(simpleclass        plan,
                            BFarray const* in,
                            BFarray*       out);
BFstatus SimpleClassDestroy(simpleclass plan);

#ifdef __cplusplus
} // extern "C"
#endif
