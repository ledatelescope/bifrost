/*
 * Copyright (c) 2018, The Bifrost Authors. All rights reserved.
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

/* 

Implements the Romein convolutional algorithm onto a GPU using CUDA. 

*/
#include <iostream>
#include <bifrost/orville.h>
#include "romein_kernels.cuh"

#include "assert.hpp"
#include "trace.hpp"
#include "utils.hpp"
#include "workspace.hpp"
#include "cuda.hpp"
#include "cuda/stream.hpp"

#include "Complex.hpp"

#include <thrust/device_vector.h>

// Maximum number of w planes to allow
#define ORVILLE_MAX_W_PLANES  128

// Use bilinear interpolation for the gridding kernel rather than nearest neighbor
#define ORVILLE_USE_KERNEL_INTERP

struct __attribute__((aligned(1))) nibble2 {
    // Yikes!  This is dicey since the packing order is implementation dependent!  
    signed char y:4, x:4;
};

struct __attribute__((aligned(1))) blenib2 {
    // Yikes!  This is dicey since the packing order is implementation dependent!
    signed char x:4, y:4;
};


template<typename RealType>
__host__ __device__
inline Complex<RealType> Complexfcma(Complex<RealType> x, Complex<RealType> y, Complex<RealType> d) {
    RealType real_res;
    RealType imag_res;
    
    real_res = (x.x *  y.x) + d.x;
    imag_res = (x.x *  y.y) + d.y;
            
    real_res =  (x.y * y.y) + real_res;  
    imag_res = -(x.y * y.x) + imag_res;          
     
    return Complex<RealType>(real_res, imag_res);
}


template<typename OutType>
__device__
inline OutType fetch_cf(cudaTextureObject_t tex,
                        int                 idx) {
    return OutType(0);
}

template<>
__device__
inline Complex32 fetch_cf<Complex32>(cudaTextureObject_t tex,
                                     int                 idx) {
    float2 p = tex1Dfetch<float2>(tex, idx);
    return Complex32(p.x, p.y);
}

template<>
__device__
inline Complex64 fetch_cf<Complex64>(cudaTextureObject_t tex,
                                     int                 idx) {
    uint4 p = tex1Dfetch<uint4>(tex, idx);
    return Complex64(__hiloint2double(p.x, p.y),
                     __hiloint2double(p.z, p.w));
}
    

template<typename OutType, typename IndexType>
__device__
inline OutType WeightAndInterp2D(OutType             w,
                                 cudaTextureObject_t kernel_tex, 
                                 IndexType           x, 
                                 IndexType           y, 
                                 int                 size) {
    OutType kx0 = fetch_cf<OutType>(kernel_tex, (int) x);
    OutType kx1 = fetch_cf<OutType>(kernel_tex, (int) x + 1);
    OutType ky0 = fetch_cf<OutType>(kernel_tex, (int) y);
    OutType ky1 = fetch_cf<OutType>(kernel_tex, (int) y + 1);
    
    OutType kx = ((int)x + 1 - x)*kx0 + (x - (int)x)*kx1;
    OutType ky = ((int)y + 1 - y)*ky0 + (y - (int)y)*ky1;
    
    return w*kx*ky;
}


template<typename OutType, typename IndexType>
__device__
inline OutType WeightAndNearestNeighbor2D(OutType             w,
                                          cudaTextureObject_t kernel_tex, 
                                          IndexType           x, 
                                          IndexType           y, 
                                          int                 size) {
    OutType kxnn = fetch_cf<OutType>(kernel_tex, (int) round(x));
    OutType kynn = fetch_cf<OutType>(kernel_tex, (int) round(y));
    
    return w*kxnn*kynn;
}


template<typename InType, typename OutType>
__global__ void orville_kernel_spln(int                         nbaseline,
                                    int                         npol,
                                    int                         maxsupport, 
                                    int                         oversample,
                                    int                         gridsize, 
                                    double                      gridres,
                                    int                         nbatch,
                                    int                         nplane,
                                    const float* __restrict__   planes,
                                    const float* __restrict__   x,
                                    const float* __restrict__   y,
                                    const float* __restrict__   z,
                                    const OutType* __restrict__ weight,
                                    cudaTextureObject_t         kernel_tex,
                                    const InType* __restrict__  d_in,
                                    OutType*                    d_out) {
    int batch_no = blockIdx.x;
    int pol_no = threadIdx.y;
    int vi_s = batch_no*nbaseline*npol+pol_no;
    int grid_s = batch_no*npol*gridsize*gridsize + pol_no*gridsize*gridsize;
    
    extern __shared__ float shared[];
    
    float* zplanes = shared;
    
    for(int i = threadIdx.x; i < nplane+1; i += blockDim.x){
        zplanes[i] = planes[i];
    }

    __syncthreads();
    
    for(int i = threadIdx.x; i < maxsupport * maxsupport; i += blockDim.x) {
        int myU = i % maxsupport;
        int myV = i / maxsupport;
        
        int grid_point_u = myU;
        int grid_point_v = myV;
        int grid_point_p = 0;
        OutType sum = OutType(0.0, 0.0);

        int vi, plane, ci_s;
        vi = plane = 0;
        ci_s = (vi + vi_s) / npol;
        while(plane < nplane) {
            if( z[ci_s] >= zplanes[plane] && z[ci_s] < zplanes[plane+1] ) {
                break;
            }
            plane++;
        }
        
        for(vi = 0; vi < (nbaseline*npol); vi+=npol) {
            ci_s = (vi + vi_s) / npol;
            //if( x[ci_s]/gridres < -gridsize/2 ) continue;
            //if( x[ci_s]/gridres > gridsize/2 ) continue;
            //if( y[ci_s]/gridres < -gridsize/2 ) continue;
            //if( y[ci_s]/gridres > gridsize/2 ) continue;
            float xf = x[ci_s]/gridres + gridsize/2 - maxsupport/2 + (1 - maxsupport%2);
            int xl = (int) round(xf);
            float yf = y[ci_s]/gridres + gridsize/2 - maxsupport/2 + (1 - maxsupport%2);
            int yl = (int) round(yf);
            
            // Find the correct z-plane to grid to
            if( z[ci_s] < zplanes[plane] || z[ci_s] >= zplanes[plane+1] ) {
                plane = 0;
                while(plane < nplane) {
                    if( z[ci_s] >= zplanes[plane] && z[ci_s] < zplanes[plane+1] ) {
                        break;
                    }
                    plane++;
                }
            }
            
            // Determine convolution point. This is basically just an
            // optimised way to calculate.
            //int myConvU = myU - u;
            //int myConvV = myV - v;
            int myConvU = 0;
            int myConvV = 0;
            if( maxsupport > 1 ) {
                myConvU = (xl - myU) % maxsupport;
                myConvV = (yl - myV) % maxsupport;    
                if (myConvU < 0) myConvU += maxsupport;
                if (myConvV < 0) myConvV += maxsupport;
            }
            float myConvUO = myConvU*oversample - ((xf - xl)*oversample) + oversample/2 - (1 - oversample%2);
            float myConvVO = myConvV*oversample - ((yf - yl)*oversample) + oversample/2 - (1 - oversample%2);
            if (myConvUO < 0) myConvUO += maxsupport*oversample;
            if (myConvVO < 0) myConvVO += maxsupport*oversample;
            
            // Determine grid point. Because of the above we know here that
            //   myGridU % max_supp = myU
            //   myGridV % max_supp = myV
            int myGridU = xl + myConvU - gridsize/2;
            int myGridV = yl + myConvV - gridsize/2;
//             if( myGridU <= -gridsize/2 ) continue;
//             if( myGridV <= -gridsize/2 ) continue;
//             if( myGridU >= gridsize/2 ) continue;
//             if( myGridU >= gridsize/2 ) continue;
            //if( myGridU + gridsize/2 >= gridsize - maxsupport/2 ) continue;
            //if( myGridV + gridsize/2 >= gridsize - maxsupport/2 ) continue;
            if( myGridU < 0 ) myGridU += gridsize;
            if( myGridV < 0 ) myGridV += gridsize;
            
            // Grid point changed?
            if (myGridU == grid_point_u && myGridV == grid_point_v && plane == grid_point_p) {
                // Nothin'
            } else {
                // Atomically add to grid. This is the bottleneck of this kernel.
                if( grid_point_u >= 0 && grid_point_u < gridsize && \
                    grid_point_v >= 0 && grid_point_v < gridsize ) {
                    atomicAdd(&d_out[grid_point_p*nbatch*npol*gridsize*gridsize + grid_s + gridsize*grid_point_v + grid_point_u].x, sum.x);
                    atomicAdd(&d_out[grid_point_p*nbatch*npol*gridsize*gridsize + grid_s + gridsize*grid_point_v + grid_point_u].y, sum.y);
                }
                // Switch to new point
                sum = OutType(0.0, 0.0);
                grid_point_u = myGridU;
                grid_point_v = myGridV;
                grid_point_p = plane;
            }
            
            //TODO: Re-do the w-kernel/gcf for our data.
#ifdef ORVILLE_USE_KERNEL_INTERP
            OutType px = WeightAndInterp2D(weight[vi+vi_s], kernel_tex, myConvVO, myConvUO, maxsupport * oversample);
#else
            OutType px = WeightAndNearestNeighbor2D(weight[vi+vi_s], kernel_tex, myConvVO, myConvUO, maxsupport * oversample);
#endif
            // Sum up
            InType temp = d_in[vi+vi_s];
            OutType vi_v = OutType(temp.x, temp.y);
            sum = Complexfcma(px, vi_v, sum);
        }
        
        if( grid_point_u >= 0 && grid_point_u < gridsize && \
            grid_point_v >= 0 && grid_point_v < gridsize ) {
            atomicAdd(&d_out[grid_point_p*nbatch*npol*gridsize*gridsize + grid_s + gridsize*grid_point_v + grid_point_u].x, sum.x);
            atomicAdd(&d_out[grid_point_p*nbatch*npol*gridsize*gridsize + grid_s + gridsize*grid_point_v + grid_point_u].y, sum.y);
        }
    }
}


template<typename InType, typename OutType>
__global__ void orville_kernel_sloc(int                         nbaseline,
                                    int                         npol,
                                    int                         maxsupport, 
                                    int                         oversample,
                                    int                         gridsize, 
                                    double                      gridres,
                                    int                         nbatch,
                                    int                         nplane,
                                    const float* __restrict__   planes,
                                    const float* __restrict__   x,
                                    const float* __restrict__   y,
                                    const float* __restrict__   z,
                                    const OutType* __restrict__ weight,
                                    cudaTextureObject_t         kernel_tex,
                                    const InType* __restrict__  d_in,
                                    OutType*                    d_out) {
    int batch_no = blockIdx.x;
    int pol_no = threadIdx.y;
    int vi_s = batch_no*nbaseline*npol+pol_no;
    int grid_s = batch_no*npol*gridsize*gridsize + pol_no*gridsize*gridsize;
    
    extern __shared__ float shared[];
    
    float* xdata = shared;
    float* ydata = xdata + nbaseline * npol;
    float* zdata = ydata + nbaseline * npol;

    for(int i = threadIdx.x; i < nbaseline; i += blockDim.x){
        xdata[i*npol + pol_no] = x[(vi_s + npol * i) / npol];
        ydata[i*npol + pol_no] = y[(vi_s + npol * i) / npol];
        zdata[i*npol + pol_no] = z[(vi_s + npol * i) / npol];
    }

    __syncthreads();
    
    for(int i = threadIdx.x; i < maxsupport * maxsupport; i += blockDim.x) {
        int myU = i % maxsupport;
        int myV = i / maxsupport;
        
        int grid_point_u = myU;
        int grid_point_v = myV;
        int grid_point_p = 0;
        OutType sum = OutType(0.0, 0.0);

        int vi, plane;
        vi = plane = 0;
        while(plane < nplane) {
            if( z[vi+vi_s] >= planes[plane] && z[vi+vi_s] < planes[plane+1] ) {
                break;
            }
            plane++;
        }
        
        for(vi = 0; vi < (nbaseline*npol); vi+=npol) {
            float xf = xdata[vi+pol_no]/gridres + gridsize/2 - maxsupport/2 + (1 - maxsupport%2);
            int xl = (int) (xf);
            float yf = ydata[vi+pol_no]/gridres + gridsize/2 - maxsupport/2 + (1 - maxsupport%2);
            int yl = (int) (yf);
            
            // Find the correct z-plane to grid to
            if( z[vi+pol_no] < planes[plane] || z[vi+pol_no] >= planes[plane+1] ) {
                plane = 0;
                while(plane < nplane) {
                    if( zdata[vi+pol_no] >= planes[plane] && zdata[vi+pol_no] < planes[plane+1] ) {
                        break;
                    }
                    plane++;
                }
            }
            
            // Determine convolution point. This is basically just an
            // optimised way to calculate.
            //int myConvU = myU - u;
            //int myConvV = myV - v;
            int myConvU = 0;
            int myConvV = 0;
            if( maxsupport > 1 ) {
                myConvU = (xl - myU) % maxsupport;
                myConvV = (yl - myV) % maxsupport;    
                if (myConvU < 0) myConvU += maxsupport;
                if (myConvV < 0) myConvV += maxsupport;
            }
            float myConvUO = myConvU*oversample - ((xf - xl)*oversample) + oversample/2 - (1 - oversample%2);
            float myConvVO = myConvV*oversample - ((yf - yl)*oversample) + oversample/2 - (1 - oversample%2);
            if (myConvUO < 0) myConvUO += maxsupport*oversample;
            if (myConvVO < 0) myConvVO += maxsupport*oversample;
            
            // Determine grid point. Because of the above we know here that
            //   myGridU % max_supp = myU
            //   myGridV % max_supp = myV
            int myGridU = xl + myConvU - gridsize/2;
            int myGridV = yl + myConvV - gridsize/2;
            if( myGridU < 0 ) myGridU += gridsize;
            if( myGridV < 0 ) myGridV += gridsize;
            
            // Grid point changed?
            if (myGridU == grid_point_u && myGridV == grid_point_v && plane == grid_point_p) {
                // Nothin'
            } else {
                // Atomically add to grid. This is the bottleneck of this kernel.
                if( grid_point_u >= 0 && grid_point_u < gridsize && \
                    grid_point_v >= 0 && grid_point_v < gridsize ) {
                    atomicAdd(&d_out[grid_point_p*nbatch*npol*gridsize*gridsize + grid_s + gridsize*grid_point_v + grid_point_u].x, sum.x);
                    atomicAdd(&d_out[grid_point_p*nbatch*npol*gridsize*gridsize + grid_s + gridsize*grid_point_v + grid_point_u].y, sum.y);
                }
                // Switch to new point
                sum = OutType(0.0, 0.0);
                grid_point_u = myGridU;
                grid_point_v = myGridV;
                grid_point_p = plane;
            }
            
            //TODO: Re-do the w-kernel/gcf for our data.
#ifdef ORVILLE_USE_KERNEL_INTERP
            OutType px = WeightAndInterp2D(weight[vi+vi_s], kernel_tex, myConvVO, myConvUO, maxsupport * oversample);
#else
            OutType px = WeightAndNearestNeighbor2D(weight[vi+vi_s], kernel_tex, myConvVO, myConvUO, maxsupport * oversample);
#endif
            // Sum up
            InType temp = d_in[vi+vi_s];
            OutType vi_v = OutType(temp.x, temp.y);
            sum = Complexfcma(px, vi_v, sum);
        }
        
        if( grid_point_u >= 0 && grid_point_u < gridsize && \
            grid_point_v >= 0 && grid_point_v < gridsize ) {
            atomicAdd(&d_out[grid_point_p*nbatch*npol*gridsize*gridsize + grid_s + gridsize*grid_point_v + grid_point_u].x, sum.x);
            atomicAdd(&d_out[grid_point_p*nbatch*npol*gridsize*gridsize + grid_s + gridsize*grid_point_v + grid_point_u].y, sum.y);
        }
    }
}


template<typename InType, typename OutType>
inline void launch_orville_kernel(int      nbaseline,
                                  int      npol,
                                  bool     polmajor,
                                  int      maxsupport, 
                                  int      oversample,
                                  int      gridsize, 
                                  double   gridres,
                                  int      nbatch,
                                  int      nplane,
                                  float*   planes,
                                  float*   xpos,
                                  float*   ypos,
                                  float*   zpos,
                                  OutType* weight,
                                  OutType* kernel,
                                  InType*  d_in,
                                  OutType* d_out,
                                  cudaStream_t stream=0) {
    //cout << "LAUNCH for " << nelement << endl;
    dim3 block(8,1);
    dim3 grid(nbatch*npol,1);
    if( polmajor ) {
        npol = 1;
    } else {
        block.y = npol;
        grid.x = nbatch;
    }
    /*
    cout << "  Block size is " << block.x << " by " << block.y << endl;
    cout << "  Grid  size is " << grid.x << " by " << grid.y << endl;
    */
    
    // Determine how to create the texture object
    // NOTE:  Assumes some type of complex float
    cudaChannelFormatKind channel_format = cudaChannelFormatKindFloat;
    int dx = 32;
    int dy = 32;
    int dz = 0;
    int dw = 0;
    if( sizeof(OutType) == sizeof(Complex64) ) {
        channel_format = cudaChannelFormatKindUnsigned;
        dz = 32;
        dw = 32;
    }
    
    // Create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = kernel;
    resDesc.res.linear.desc.f = channel_format;
    resDesc.res.linear.desc.x = dx;
    resDesc.res.linear.desc.y = dy;
    resDesc.res.linear.desc.z = dz;
    resDesc.res.linear.desc.w = dw;
    resDesc.res.linear.sizeInBytes = maxsupport*oversample*sizeof(OutType);
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    
    cudaTextureObject_t kernel_tex;
    BF_CHECK_CUDA_EXCEPTION(cudaCreateTextureObject(&kernel_tex, &resDesc, &texDesc, NULL),
                            BF_STATUS_INTERNAL_ERROR);
    
    void* args[] = {&nbaseline,
                    &npol,
                    &maxsupport,
                    &oversample,
                    &gridsize, 
                    &gridres,
                    &nbatch,
                    &nplane,
                    &planes,
                    &xpos,
                    &ypos,
                    &zpos,
                    &weight,
                    &kernel_tex,
                    &d_in,
                    &d_out};
    size_t loc_size = 3 * nbaseline * npol * sizeof(float);
    size_t shared_mem_size = 16384; //Just keep this vanilla for now
    if(loc_size <= shared_mem_size) {
        BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)orville_kernel_sloc<InType,OutType>,
                                                 grid, block,
                                                 &args[0], 3*nbaseline*npol*sizeof(float), stream),
                                BF_STATUS_INTERNAL_ERROR);
    } else {
        BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)orville_kernel_spln<InType,OutType>,
                                                 grid, block,
                                                 &args[0], (nplane+1)*sizeof(float), stream),
                                BF_STATUS_INTERNAL_ERROR);
    }
    
    BF_CHECK_CUDA_EXCEPTION(cudaDestroyTextureObject(kernel_tex),
                            BF_STATUS_INTERNAL_ERROR);
}
                   

int compare_z_values(const void* a, const void* b) {
    if( *(float*)a < *(float*)b ) return -1;
    if( *(float*)a > *(float*)b ) return  1;
    return 0;
}

template<typename InType>
InType signed_sqrt(InType x) {
    if( x < 0 ) {
        return -sqrt(-x);
    } else {
        return sqrt(x);
    }
}

class BForville_impl {
    typedef int    IType;
    typedef double FType;
public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
    typedef float  DType;
private:
    IType        _ntimechan;
    IType        _nbaseline;
    IType        _npol;
    bool         _polmajor;
    IType        _maxsupport;
    IType        _oversample;
    IType        _gridsize;
    FType        _gridres;
    FType        _gridwres;
    IType        _nxyz = 0;
    float*       _x = NULL;
    float*       _y = NULL;
    float*       _z = NULL;
    void*        _weight = NULL;
    BFdtype      _tkernel = BF_DTYPE_INT_TYPE;
    void*        _kernel = NULL;
    IType        _nplane = 0;
    float*       _planes = NULL;
    float*       _midpoints = NULL;
    IType        _plan_stride;
    Workspace    _plan_storage;
    // TODO: Use something other than Thrust
    thrust::device_vector<char> _dv_plan_storage;
    cudaStream_t _stream;
public:
    BForville_impl() : _ntimechan(1), _nbaseline(1), _npol(1), _polmajor(true), \
                      _maxsupport(1), _oversample(1), _stream(g_cuda_stream) {}
    inline IType ntimechan()  const { return _ntimechan;  }
    inline IType nbaseline()  const { return _nbaseline;  }
    inline IType npol()       const { return _npol;       }
    inline bool polmajor()    const { return _polmajor;   }
    inline IType maxsupport() const { return _maxsupport; }
    inline IType oversample() const { return _oversample; }
    inline IType gridsize()   const { return _gridsize;   }
    inline FType gridres()    const { return _gridres;    }
    inline FType grdiwres()   const { return _gridwres;   }
    inline IType nxyz()       const { return _nxyz;       }
    inline IType nplane()     const { return _nplane;     }
    inline IType tkernel()    const { return _tkernel;    }
    void init(IType ntimechan,
              IType nbaseline,
              IType npol,
              bool  polmajor,
              IType maxsupport, 
              IType oversample,
              IType gridsize,
              FType gridres,
              FType gridwres) {
        BF_TRACE();
        _ntimechan  = ntimechan;
        _nbaseline  = nbaseline;
        _npol       = npol;
        _polmajor   = polmajor;
        _maxsupport = maxsupport;
        _oversample = oversample;
        _gridsize   = gridsize + maxsupport;
        // Find an optimal grid size that allows for at least 'maxsupport' padding
        // NOTE:  "Optimal" here means that is can be factored into 2, 3, 5, and 7,
        //        exclusively.
        IType i, value;
        bool found_opt_gridsize = false;
        for(i=_gridsize; i<(IType) round(_gridsize*1.15); i++) {
            value = i;
            while( value > 1 ) {
                if( value % 2 == 0 ) {
                    value /= 2;
                } else if( value % 3 == 0 ) {
                    value /= 3;
                } else if( value % 5 == 0 ) {
                    value /= 5;
                } else if( value % 7 == 0 ) {
                    value /= 7;
                } else {
                    break;
                }
            }
            if( value == 1 ) {
                found_opt_gridsize = true;
                _gridsize = i;
                break;
            }
        }
        BF_ASSERT_EXCEPTION(found_opt_gridsize, BF_STATUS_INVALID_SHAPE);
        _gridres    = 1.0 / (2.0*_gridsize) / sin(gridres*M_PI/180.0/2.0);
        _gridwres   = gridwres;
    }
    bool init_plan_storage(void* storage_ptr, BFsize* storage_size) {
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        enum {
            ALIGNMENT_BYTES = 512,
            ALIGNMENT_ELMTS = ALIGNMENT_BYTES / sizeof(float)
        };
        Workspace workspace(ALIGNMENT_BYTES);
        _plan_stride = round_up(ORVILLE_MAX_W_PLANES+1, ALIGNMENT_ELMTS);
        workspace.reserve(ORVILLE_MAX_W_PLANES+1, &_planes);
        workspace.reserve(ORVILLE_MAX_W_PLANES+1, &_midpoints);
        if( storage_size ) {
            if( !storage_ptr ) {
                // Return required storage size
                *storage_size = workspace.size();
                return false;
            } else {
                BF_ASSERT_EXCEPTION(*storage_size >= workspace.size(),
                                    BF_STATUS_INSUFFICIENT_STORAGE);
            }
        } else {
            // Auto-allocate storage
            BF_ASSERT_EXCEPTION(!storage_ptr, BF_STATUS_INVALID_ARGUMENT);
            _dv_plan_storage.resize(workspace.size());
            storage_ptr = thrust::raw_pointer_cast(&_dv_plan_storage[0]);
        }
        workspace.commit(storage_ptr);
        
        this->reset_state();
        return true;
    }
    void set_positions(BFarray const* positions) { 
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        BF_ASSERT_EXCEPTION(positions->dtype == BF_DTYPE_F32, BF_STATUS_UNSUPPORTED_DTYPE);
        
        int npositions = positions->shape[1];
        int nbl = positions->shape[positions->ndim-1];
        int stride = positions->shape[1];
        for(int i=2; i<positions->ndim-1; ++i) {
            npositions *= positions->shape[i];
            stride *= positions->shape[i];
        }
        stride *= positions->shape[positions->ndim-1];
        _nxyz = npositions;
        _x = (float *) positions->data;
        _y = _x + stride;
        _z = _y + stride;
        
        float* zsorted;
        zsorted = (float*) malloc(npositions*nbl*sizeof(float));
        
        BF_CHECK_CUDA_EXCEPTION(cudaMemcpyAsync(zsorted, 
                                                _z, 
                                                npositions*nbl*sizeof(float), 
                                                cudaMemcpyDeviceToHost,
                                                _stream),
                                BF_STATUS_MEM_OP_FAILED);
        
        BF_CHECK_CUDA_EXCEPTION(cudaStreamSynchronize(_stream),
                                BF_STATUS_DEVICE_ERROR );
        
        qsort(zsorted, npositions*nbl, sizeof(float), &compare_z_values);
        /*
        std::cout << npositions*nbl << " z values range from " << *(zsorted+0) << " to " << *(zsorted+npositions-1) << std::endl;
        */
        
        _nplane = 1;
        int count = 1;
        float* planes;
        float* midpoints;
        planes = (float*) malloc((ORVILLE_MAX_W_PLANES+1)*sizeof(float));
        midpoints = (float*) malloc((ORVILLE_MAX_W_PLANES+1)*sizeof(float));
        *(planes + 0) = *(zsorted + 0);
        *(midpoints + 0) = *(zsorted + 0);
        for(int i=1; i<npositions*nbl; i++) {
            if( signed_sqrt(*(zsorted + i)) > signed_sqrt(*(planes + _nplane - 1)) + _gridwres ) {
                *(midpoints + _nplane - 1) /= count;
                count = 0;
                
                _nplane++;
                BF_ASSERT_EXCEPTION(_nplane < ORVILLE_MAX_W_PLANES, BF_STATUS_INTERNAL_ERROR);
                *(planes + _nplane - 1) = *(zsorted + i);
                *(midpoints + _nplane - 1) = *(zsorted + i);
                count++;
            } else {
                *(midpoints + _nplane - 1) += *(zsorted + i);
                count++;
            }
        }
        *(planes + _nplane) = *(zsorted + npositions*nbl - 1) + _gridwres;
        *(midpoints + _nplane - 1) += *(zsorted + npositions*nbl - 1);
        count++;
        *(midpoints + _nplane - 1) /= count;
        
        /* std::cout << "Planes are:" << std::endl;
        for(int i=0; i<_nplane; i++) {
            std::cout << "  " << i << ": " << *(planes + i) << " to " << *(planes + i + 1) \
                      << " with " << *(midpoints + i) << std::endl;
        } */
        
        BF_CHECK_CUDA_EXCEPTION(cudaMemcpyAsync(_planes, 
                                                planes, 
                                                (ORVILLE_MAX_W_PLANES+1)*sizeof(float), 
                                                cudaMemcpyHostToDevice,
                                                _stream),
                                BF_STATUS_MEM_OP_FAILED);
        BF_CHECK_CUDA_EXCEPTION(cudaMemcpyAsync(_midpoints, 
                                                midpoints, 
                                                (ORVILLE_MAX_W_PLANES+1)*sizeof(float), 
                                                cudaMemcpyHostToDevice,
                                                _stream),
                                BF_STATUS_MEM_OP_FAILED);
        
        BF_CHECK_CUDA_EXCEPTION(cudaStreamSynchronize(_stream),
                                BF_STATUS_DEVICE_ERROR );
        
        free(zsorted);
        free(planes);
        free(midpoints);
    }
    void get_midpoints(BFarray const* midpoints) {
        BF_CHECK_CUDA_EXCEPTION(cudaMemcpyAsync(midpoints->data,
                                                _midpoints,
                                                _nplane*sizeof(float),
                                                cudaMemcpyDeviceToHost,
                                                _stream),
                                BF_STATUS_MEM_OP_FAILED);
        
        BF_CHECK_CUDA_EXCEPTION( cudaStreamSynchronize(_stream),
                                 BF_STATUS_DEVICE_ERROR );
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
    }
    void set_weights(BFarray const* weights) {
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        BF_ASSERT_EXCEPTION(weights->dtype == BF_DTYPE_CF32 \
                                              || BF_DTYPE_CF64, BF_STATUS_UNSUPPORTED_DTYPE);
        
        if( _kernel == NULL ) {
            _tkernel = weights->dtype;
        }
        _weight = (void*) weights->data;
    }
    void set_kernel(BFarray const* kernel) {
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        BF_ASSERT_EXCEPTION(kernel->dtype == BF_DTYPE_CF32 \
                                             || BF_DTYPE_CF64, BF_STATUS_UNSUPPORTED_DTYPE);
        
        if( _weight == NULL ) {
            _tkernel = kernel->dtype;
        }
        _kernel = (void*) kernel->data;
    }
    void reset_state() {
        BF_ASSERT_EXCEPTION(_planes != NULL,  BF_STATUS_INVALID_STATE);
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
        
        // Reset the state
        _nplane = 0;
        BF_CHECK_CUDA_EXCEPTION( cudaMemsetAsync(_planes,
                                                0,
                                                sizeof(float)*(ORVILLE_MAX_W_PLANES+1),
                                                _stream),
                                BF_STATUS_MEM_OP_FAILED );
        BF_CHECK_CUDA_EXCEPTION( cudaMemsetAsync(_midpoints,
                                                0,
                                                sizeof(float)*(ORVILLE_MAX_W_PLANES+1),
                                                _stream),
                                BF_STATUS_MEM_OP_FAILED );
        
        BF_CHECK_CUDA_EXCEPTION( cudaStreamSynchronize(_stream),
                                 BF_STATUS_DEVICE_ERROR );
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
    }
    void execute(BFarray const* in, BFarray const* out) {
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        BF_ASSERT_EXCEPTION(_x != NULL,      BF_STATUS_INVALID_STATE);
        BF_ASSERT_EXCEPTION(_y != NULL,      BF_STATUS_INVALID_STATE);
        BF_ASSERT_EXCEPTION(_z != NULL,      BF_STATUS_INVALID_STATE);
        BF_ASSERT_EXCEPTION(_weight != NULL, BF_STATUS_INVALID_STATE);
        BF_ASSERT_EXCEPTION(_kernel != NULL, BF_STATUS_INVALID_STATE);
        BF_ASSERT_EXCEPTION(out->dtype == BF_DTYPE_CF32 \
                                          || BF_DTYPE_CF64, BF_STATUS_UNSUPPORTED_DTYPE);
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
        
        int nbatch = in->shape[0];
        
#define LAUNCH_ORVILLE_KERNEL(IterType,OterType) \
        launch_orville_kernel(_nbaseline, _npol, _polmajor, _maxsupport, _oversample, \
                             _gridsize, _gridres, nbatch, _nplane, _planes, \
                             _x, _y, _z, (OterType)_weight, (OterType)_kernel, \
                             (IterType)in->data, (OterType)out->data, \
                             _stream)
        
        switch( in->dtype ) {
            case BF_DTYPE_CI4:
                if( in->big_endian ) {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_ORVILLE_KERNEL(nibble2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_ORVILLE_KERNEL(nibble2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                } else {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_ORVILLE_KERNEL(blenib2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_ORVILLE_KERNEL(blenib2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                }
                break;
            case BF_DTYPE_CI8:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ORVILLE_KERNEL(char2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_ORVILLE_KERNEL(char2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                };
                break;
            case BF_DTYPE_CI16:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ORVILLE_KERNEL(short2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_ORVILLE_KERNEL(short2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ORVILLE_KERNEL(int2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_ORVILLE_KERNEL(int2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ORVILLE_KERNEL(long2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_ORVILLE_KERNEL(long2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ORVILLE_KERNEL(float2*, Complex32*);   break;
                    case BF_DTYPE_CF64: LAUNCH_ORVILLE_KERNEL(float2*, Complex64*);   break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ORVILLE_KERNEL(double2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_ORVILLE_KERNEL(double2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        }
#undef LAUNCH_ORVILLE_KERNEL
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
    }
    cudaStream_t get_stream() {
        return _stream;
    }
    void set_stream(cudaStream_t stream) {
        _stream = stream;
    }
};

BFstatus bfOrvilleCreate(BForville* plan_ptr) {
    BF_TRACE();
    BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN_ELSE(*plan_ptr = new BForville_impl(),
                       *plan_ptr = 0);
}

BFstatus bfOrvilleInit(BForville      plan,
                       BFarray const* positions,
                       BFarray const* weights,
                       BFarray const* kernel1D,
                       BFsize         gridsize,
                       double         gridres,
                       double         gridwres,
                       BFsize         oversample,
                       BFbool         polmajor,
                       BFspace        space,
                       void*          plan_storage,
                       BFsize*        plan_storage_size) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(positions,                                BF_STATUS_INVALID_POINTER);
    BF_ASSERT(positions->ndim >= 3,                     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(positions->shape[0] == 3, BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(space_accessible_from(positions->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_ASSERT(weights,                                BF_STATUS_INVALID_POINTER);
    BF_ASSERT(weights->ndim >= 3,                     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(space_accessible_from(weights->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_ASSERT(kernel1D,                              BF_STATUS_INVALID_POINTER);
    BF_ASSERT(kernel1D->ndim == 1,                   BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(space_accessible_from(kernel1D->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_ASSERT(space_accessible_from(space, BF_SPACE_CUDA),
              BF_STATUS_UNSUPPORTED_SPACE);
    
    // Discover the dimensions of the positions/weights/kernel.
    int npositions, nweights,nbaseline, npol, maxsupport;
    npositions = positions->shape[1];
    for(int i=2; i<positions->ndim-1; ++i) {
        npositions *= positions->shape[i];
    }
    
    nweights = weights->shape[0];
    for(int i=1; i<weights->ndim-2; ++i) {
        nweights *= weights->shape[i];
    }
    if( polmajor ) {
         npol = weights->shape[weights->ndim-2];
         nbaseline = weights->shape[weights->ndim-1];
    } else {
        nbaseline = weights->shape[weights->ndim-2];
        npol = weights->shape[weights->ndim-1];
    }
    
    std::cout << "npositions: " << npositions << std::endl;
    std::cout << "nbaseline:  " << nbaseline << std::endl;
    std::cout << "npol:       " << npol << std::endl;
    std::cout << "nweights:   " << nweights << std::endl;
    
    maxsupport = kernel1D->shape[kernel1D->ndim-1] / oversample;
    
    // Validate
    BF_ASSERT(npositions == nweights, BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(positions->shape[positions->ndim-1] == nbaseline, BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(maxsupport % 2 == 1, BF_STATUS_UNSUPPORTED_SHAPE);
//     BF_ASSERT(oversample % 2 == 1, BF_STATUS_UNSUPPORTED_SHAPE);
    
    BF_TRY(plan->init(npositions, nbaseline, npol, polmajor, maxsupport, oversample, gridsize, gridres, gridwres));
    BF_TRY(plan->init_plan_storage(plan_storage, plan_storage_size));
    BF_TRY(plan->set_positions(positions));
    BF_TRY(plan->set_weights(weights));
    BF_TRY_RETURN(plan->set_kernel(kernel1D));
}
BFstatus bfOrvilleGetStream(BForville plan,
                            void*     stream) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN(*(cudaStream_t*)stream = plan->get_stream());
}
BFstatus bfOrvilleSetStream(BForville    plan,
                            void const* stream) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}
BFstatus bfOrvilleSetPositions(BForville      plan,
                               BFarray const* positions) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(positions,            BF_STATUS_INVALID_POINTER);
    BF_ASSERT(positions->ndim >= 3, BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(positions->shape[positions->ndim-2] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(positions->shape[0] == 3,                 BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(space_accessible_from(positions->space,   BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    
    BF_TRY_RETURN(plan->set_positions(positions));
}
BFstatus bfOrvilleGetProjectionSetup(BForville plan,
                                     int* ntimechan,
                                     int* gridsize,
                                     int* npol,
                                     int* nplane,
                                     BFarray const* midpoints) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(midpoints,            BF_STATUS_INVALID_POINTER);
    BF_ASSERT(midpoints->ndim == 1, BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(midpoints->shape[0] >= plan->nplane(), BF_STATUS_INVALID_SHAPE);
    
    *ntimechan = plan->ntimechan();
    *gridsize = plan->gridsize();
    *npol = plan->npol();
    *nplane = plan->nplane();
    
    BF_TRY_RETURN(plan->get_midpoints(midpoints));
}
BFstatus bfOrvilleSetWeights(BForville      plan,
                             BFarray const* weights) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(weights,            BF_STATUS_INVALID_POINTER);
    BF_ASSERT(weights->ndim >= 3, BF_STATUS_INVALID_SHAPE  );
    if( plan->polmajor() ) {
        BF_ASSERT(weights->shape[weights->ndim-2] == plan->npol(),      BF_STATUS_INVALID_SHAPE  );
        BF_ASSERT(weights->shape[weights->ndim-1] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE  );
    } else {
        BF_ASSERT(weights->shape[weights->ndim-2] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE  );
        BF_ASSERT(weights->shape[weights->ndim-1] == plan->npol(),      BF_STATUS_INVALID_SHAPE  );
    }
    BF_ASSERT(space_accessible_from(weights->space,   BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    
    if( plan->tkernel() != BF_DTYPE_INT_TYPE) {
        BF_ASSERT(weights->dtype == plan->tkernel(),    BF_STATUS_UNSUPPORTED_DTYPE);
    }
    
    BF_TRY_RETURN(plan->set_weights(weights));
}
BFstatus bfOrvilleSetKernel(BForville      plan, 
                            BFarray const* kernel1D) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(kernel1D,            BF_STATUS_INVALID_POINTER);
    BF_ASSERT(kernel1D->ndim == 1, BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(kernel1D->shape[kernel1D->ndim-1] == plan->maxsupport()*plan->oversample(), \
              BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(space_accessible_from(kernel1D->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    
    if( plan->tkernel() != BF_DTYPE_INT_TYPE) {
        BF_ASSERT(kernel1D->dtype == plan->tkernel(),    BF_STATUS_UNSUPPORTED_DTYPE);
    }
    
    BF_TRY_RETURN(plan->set_kernel(kernel1D));
}
BFstatus bfOrvilleExecute(BForville      plan,
                          BFarray const* in,
                          BFarray const* out) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
    BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
    BF_ASSERT( in->ndim >= 3,          BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->ndim == in->ndim+2, BF_STATUS_INVALID_SHAPE);
    
    BFarray in_flattened;
    if( in->ndim > 3 ) {
        // Keep the last two dim but attempt to flatten all others
        unsigned long keep_dims_mask = padded_dims_mask(in);
        keep_dims_mask |= 0x1 << (in->ndim-1);
        keep_dims_mask |= 0x1 << (in->ndim-2);
        keep_dims_mask |= 0x1 << (in->ndim-3);
        flatten(in,   &in_flattened, keep_dims_mask);
        in  =  &in_flattened;
        BF_ASSERT(in_flattened.ndim == 3, BF_STATUS_UNSUPPORTED_SHAPE);
    }
    /*
    std::cout << "ndim = " << in->ndim << std::endl;
    std::cout << "   0 = " << in->shape[0] << std::endl;
    std::cout << "   1 = " << in->shape[1] << std::endl;
    std::cout << "   2 = " << in->shape[2] << std::endl;
    */
    BF_ASSERT( in->shape[0] == plan->nxyz(),     BF_STATUS_INVALID_SHAPE);
    if( plan->polmajor() ) {
        BF_ASSERT( in->shape[1] == plan->npol(),      BF_STATUS_INVALID_SHAPE);
        BF_ASSERT( in->shape[2] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE);
    } else {
        BF_ASSERT( in->shape[1] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE);
        BF_ASSERT( in->shape[2] == plan->npol(),      BF_STATUS_INVALID_SHAPE);
    }
    
    BFarray out_flattened;
    if( out->ndim > 4 ) {
        // Keep the last three dim but attempt to flatten all others
        unsigned long keep_dims_mask = padded_dims_mask(out);
        keep_dims_mask |= 0x1;
        keep_dims_mask |= 0x1 << (out->ndim-1);
        keep_dims_mask |= 0x1 << (out->ndim-2);
        keep_dims_mask |= 0x1 << (out->ndim-3);
        keep_dims_mask |= 0x1 << (out->ndim-4);
        flatten(out,   &out_flattened, keep_dims_mask);
        out  =  &out_flattened;
        BF_ASSERT(out_flattened.ndim == 5, BF_STATUS_UNSUPPORTED_SHAPE);
    }
    /*
    std::cout << "ndim = " << out->ndim << std::endl;
    std::cout << "   0 = " << out->shape[0] << std::endl;
    std::cout << "   1 = " << out->shape[1] << std::endl;
    std::cout << "   2 = " << out->shape[2] << std::endl;
    std::cout << "   3 = " << out->shape[3] << std::endl;
    */
    BF_ASSERT(out->shape[0] == plan->nplane(),   BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[1] == plan->nxyz(),     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[2] == plan->npol(),     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[3] == plan->gridsize(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[4] == plan->gridsize(), BF_STATUS_INVALID_SHAPE);
    
    BF_ASSERT(out->dtype == plan->tkernel(),    BF_STATUS_UNSUPPORTED_DTYPE);
    
    BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_TRY_RETURN(plan->execute(in, out));
}

BFstatus bfOrvilleDestroy(BForville plan) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    delete plan;
    return BF_STATUS_SUCCESS;
}
