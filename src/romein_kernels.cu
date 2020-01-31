#include "romein_kernels.cuh"

/*****************************
        Device Functions
 *****************************/

//From Kyrills implementation in SKA/RC
__device__ void scatter_grid_add(cuComplex *uvgrid,
				 int grid_size,
				 int grid_pitch,
				 int grid_point_u,
				 int grid_point_v,
				 cuComplex sum){

    if (grid_point_u < 0 || grid_point_u >= grid_size ||
      grid_point_v < 0 || grid_point_v >= grid_size)
    return;

    // Add to grid. This is the bottleneck of the entire kernel
    atomicAdd(&uvgrid[grid_point_u + grid_pitch*grid_point_v].x, sum.x); // Re
    atomicAdd(&uvgrid[grid_point_u + grid_pitch*grid_point_v].y, sum.y); // Im
}

#ifdef __COUNT_VIS__
__device__ void scatter_grid_point(cuComplex* fdata,
				   cuComplex* uvgrid, // Our main UV Grid
				   cuComplex* illum, //Our W-Kernel
				   int* x,
				   int* y,
				   int* z,
				   int max_supp, // Max size of W-Kernel
				   int myU, //Our assigned u/v points.
				   int myV, // ^^^
				   int grid_size, //The size of our w-towers subgrid.
				   int data_size,
				   int batch_no,
				   unsigned long long int *visc_reg){ 
#else
__device__ void scatter_grid_point(cuComplex* fdata, //Our fourier data
				   cuComplex* uvgrid, // Our main UV Grid
				   cuComplex* illum, //Our W-Kernel
				   int* x, // Ant x location data
				   int* y, // Ant y location data
				   int* z, // Ant z location data
				   int max_supp, // Max size of W-Kernel
				   int myU, //Our assigned u/v points.
				   int myV, // ^^^
				   int grid_size,
				   int data_size,
				   int batch_no){ 
#endif
  
  int grid_point_u = myU, grid_point_v = myV;
  cuComplex sum  = make_cuComplex(0.0,0.0);
  short supp = max_supp;
  int vi_s = batch_no * data_size;
  int grid_s = grid_size * grid_size * batch_no;
  int vi = 0;
  for (vi = vi_s; vi < (vi_s+data_size); ++vi){

    int u = x[vi]; 
    int v = y[vi];

    // Determine convolution point. This is basically just an
    // optimised way to calculate.
    //int myConvU = myU - u;
    //int myConvV = myV - v;
    int myConvU = (u - myU) % max_supp;
    int myConvV = (v - myV) % max_supp;    
    if (myConvU < 0) myConvU += max_supp;
    if (myConvV < 0) myConvV += max_supp;

    // Determine grid point. Because of the above we know here that
    //   myGridU % max_supp = myU
    //   myGridV % max_supp = myV
    int myGridU = u + myConvU
      , myGridV = v + myConvV;

    // Grid point changed?
    if (myGridU != grid_point_u || myGridV != grid_point_v) {
      // Atomically add to grid. This is the bottleneck of this kernel.
      scatter_grid_add(uvgrid+grid_s, grid_size, grid_size, grid_point_u, grid_point_v, sum);
      // Switch to new point
      sum = make_cuComplex(0.0, 0.0);
      grid_point_u = myGridU;
      grid_point_v = myGridV;
    }
    //TODO: Re-do the w-kernel/gcf for our data.
    //	cuDoubleComplex px;
    cuComplex px = illum[myConvV * supp + myConvU];// ??
    //cuComplex px = *(cuComplex*)&wkern->kern_by_w[w_plane].data[sub_offset + myConvV * supp + myConvU];	
    // Sum up
    cuComplex vi_v = fdata[vi];
    sum = cuCfmaf(cuConjf(px), vi_v, sum);

  }
  // Add remaining sum to grid
  #ifdef __COUNT_VIS__
  atomicAdd(visc_reg,vi);
  #endif
  scatter_grid_add(uvgrid+grid_s, grid_size, grid_size, grid_point_u, grid_point_v, sum);
}


/*******************
   Romein Kernel
 ******************/
 
#ifdef __COUNT_VIS__
 __global__ void scatter_grid_kernel(cuComplex* fdata, //Our fourier data
				     cuComplex* uvgrid, // Our main UV Grid
				     cuComplex* illum, //Our W-Kernel
				     int* x, // Ant x location data
				     int* y, // Ant y location data
				     int* z, // Ant z location data
				     int max_support, //  Convolution size
				     int grid_size, // Subgrid size
				     int data_size,
				     unsigned long long int* visc_reg){
#else
__global__ void scatter_grid_kernel(cuComplex* fdata, //Our fourier data
				    cuComplex* uvgrid, // Our main UV Grid
				    cuComplex* illum, //Our W-Kernel
				    int* x, // Ant x location data
				    int* y, // Ant y location data
				    int* z, // Ant z location data
				    int max_support, //  Convolution size
				    int grid_size,
				    int data_size){
				
#endif
  //Assign some visibilities to grid;
    int batch_no = blockIdx.x;
    for(int i = threadIdx.x; i < max_support * max_support; i += blockDim.x){
	//  int i = threadIdx.x + blockIdx.x * blockDim.x;
	int myU = i % max_support;
	int myV = i / max_support;
    
#ifdef __COUNT_VIS__
	scatter_grid_point(fdata, uvgrid, illum, x, y, z, max_support, myU, myV, grid_size, data_size, batch_no, visc_reg);
#else
	scatter_grid_point(fdata, uvgrid, illum, x, y, z, max_support, myU, myV, grid_size, data_size, batch_no);
#endif		       
  }
}

