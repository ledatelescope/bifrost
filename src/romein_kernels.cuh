//CUDA Includes
#include <hip/hip_complex.h>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "math.h"

//Romein Include
#include "bifrost/romein.h"    

/*****************************
      Device Functions
******************************/

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600


#else //Pre-pascal devices.

__device__ inline double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif

__global__ void scatter_grid_kernel(hipComplex* fdata,
				    hipComplex* uvgrid, //Our UV-Grid
				    hipComplex* illum, // Illumination Pattern
				    int* x,
				    int* y,
				    int* z,
				    int max_support, //  Convolution size
				    int grid_size,
				    int data_size);