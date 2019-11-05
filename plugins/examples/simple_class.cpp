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
