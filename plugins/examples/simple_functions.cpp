/*
 * Copyright (c) 2019, The Bifrost Authors. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of The Bifrost Authors nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
