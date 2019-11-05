#include "assert.hpp"
#include <bifrost/array.h>
#include <bifrost/common.h>
#include "utils.hpp"
#include <stdlib.h>
#include <stdio.h>

#include "simple_functions.h"

BFstatus AddStuff(BFarray *xdata, BFarray *ydata, BFarray *zdata)
{
    BF_ASSERT(xdata->dtype == BF_DTYPE_F32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(ydata->dtype == BF_DTYPE_F32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(zdata->dtype == BF_DTYPE_F32, BF_STATUS_UNSUPPORTED_DTYPE);
    
    BF_ASSERT(space_accessible_from(xdata->space, BF_SPACE_SYSTEM),
              BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(ydata->space, BF_SPACE_SYSTEM),
              BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(zdata->space, BF_SPACE_SYSTEM),
              BF_STATUS_UNSUPPORTED_SPACE);
    
    long nelements = num_contiguous_elements(xdata);

    float* x = (float *)xdata->data;
    float* y = (float *)ydata->data;
    float* z = (float *)zdata->data;

    for(int i=0; i < nelements; i +=1) {
       z[i] = x[i] + y[i];
    }

    return BF_STATUS_SUCCESS;
}

BFstatus SubtractStuff(BFarray *xdata, BFarray *ydata, BFarray *zdata)
{
    BF_ASSERT(xdata->dtype == BF_DTYPE_F32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(ydata->dtype == BF_DTYPE_F32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(zdata->dtype == BF_DTYPE_F32, BF_STATUS_UNSUPPORTED_DTYPE);
    
    BF_ASSERT(space_accessible_from(xdata->space, BF_SPACE_SYSTEM),
              BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(ydata->space, BF_SPACE_SYSTEM),
              BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(zdata->space, BF_SPACE_SYSTEM),
              BF_STATUS_UNSUPPORTED_SPACE);
    
    long nelements = num_contiguous_elements(xdata);

    float* x = (float *)xdata->data;
    float* y = (float *)ydata->data;
    float* z = (float *)zdata->data;

    for(int i=0; i < nelements; i +=1) {
       z[i] = x[i] - y[i];
    }

    return BF_STATUS_SUCCESS;
}