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
#include "utils.hpp"

#include "simple_class.h"

class simpleclass_impl {
private:
    float _value;
public:
    simpleclass_impl() {}
    inline float value() const { return _value; }
    void init(float value) {
        _value = value;
    }
    void execute(BFarray const* in, BFarray* out) {
        long nelements = num_contiguous_elements(in);
        
        float* x = (float *)in->data;
        float* y = (float *)out->data;
        
        for(int i=0; i < nelements; i +=1) {
            y[i] = x[i] + _value;
        }
    }
};

BFstatus SimpleClassCreate(simpleclass* plan_ptr) {
    BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN_ELSE(*plan_ptr = new simpleclass_impl(),
                       *plan_ptr = 0);
}

BFstatus SimpleClassInit(simpleclass plan,
                     float value) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_TRY_RETURN(plan->init(value));
}

BFstatus SimpleClassExecute(simpleclass        plan,
                        BFarray const* in,
                        BFarray*       out) {
    BF_ASSERT(in->dtype == BF_DTYPE_F32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(out->dtype == BF_DTYPE_F32, BF_STATUS_UNSUPPORTED_DTYPE);
    
    BF_ASSERT(space_accessible_from(in->space, BF_SPACE_SYSTEM),
              BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(out->space, BF_SPACE_SYSTEM),
              BF_STATUS_UNSUPPORTED_SPACE);
    
    BF_TRY_RETURN(plan->execute(in, out));
}

BFstatus SimpleClassDestroy(simpleclass plan) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    delete plan;
    return BF_STATUS_SUCCESS;
}
