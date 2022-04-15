//CUDA Includes
#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "math.h"
#include <thrust/random.h>

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

__global__ void scatter_grid_kernel(cuComplex* fdata,
				    cuComplex* uvgrid, //Our UV-Grid
				    cuComplex* illum, // Illumination Pattern
				    int* x,
				    int* y,
				    int* z,
				    int max_support, //  Convolution size
				    int grid_size,
				    int data_size);