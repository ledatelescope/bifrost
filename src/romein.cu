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
#include <bifrost/romein.h>
#include "romein_kernels.cuh"

#include "assert.hpp"
#include "trace.hpp"
#include "utils.hpp"
#include "cuda.hpp"
#include "cuda/stream.hpp"

#include "Complex.hpp"

template<typename RealType>
__host__ __device__
inline Complex<RealType> Complexfcma(Complex<float> x, Complex<RealType> y, Complex<RealType> d) {
    RealType real_res;
    RealType imag_res;
    
    real_res = (x.x *  y.x) + d.x;
    imag_res = (x.x *  y.y) + d.y;
            
    real_res =  (x.y * y.y) + real_res;  
    imag_res = -(x.y * y.x) + imag_res;          
     
    return Complex<RealType>(real_res, imag_res);
}

template<typename InType, typename OutType>
__global__ void romein_kernel(int                           nbaseline,
                              int                           maxsupport, 
                              int                           gridsize, 
                              int                           nbatch,
                              const int3* __restrict__       positions,
                              const Complex32* __restrict__ kernels,
                              const InType* __restrict__    d_in,
                              OutType*                      d_out) {
    int batch_no = blockIdx.x;
    for(int i = threadIdx.x; i < maxsupport * maxsupport; i += blockDim.x) {
        int myU = i % maxsupport;
        int myV = i / maxsupport;
        
        int grid_point_u = myU;
        int grid_point_v = myV;
        OutType sum = OutType(0.0, 0.0);
        int vi_s = batch_no*nbaseline;
        int grid_s = batch_no*gridsize*gridsize;
        int vi = 0;
        for(vi = vi_s; vi < (vi_s+nbaseline); ++vi) {
            int3 uvw = positions[vi]; 
            
            // Determine convolution point. This is basically just an
            // optimised way to calculate.
            //int myConvU = myU - u;
            //int myConvV = myV - v;
            int myConvU = 0;
            int myConvV = 0;
            if( maxsupport > 1 ) {
                myConvU = (uvw.x - myU) % maxsupport;
                myConvV = (uvw.y - myV) % maxsupport;    
                if (myConvU < 0) myConvU += maxsupport;
                if (myConvV < 0) myConvV += maxsupport;
            } 
            
            // Determine grid point. Because of the above we know here that
            //   myGridU % max_supp = myU
            //   myGridV % max_supp = myV
            int myGridU = uvw.x + myConvU;
            int myGridV = uvw.y + myConvV;
            
            // Grid point changed?
            if (myGridU == grid_point_u && myGridV == grid_point_v) {
                // Nothin'
            } else {
                // Atomically add to grid. This is the bottleneck of this kernel.
                if( grid_point_u >= 0 && grid_point_u < gridsize && \
                    grid_point_v >= 0 && grid_point_v < gridsize ) {
                    atomicAdd(&d_out[grid_s + gridsize*grid_point_v + grid_point_u].x, sum.x);
                    atomicAdd(&d_out[grid_s + gridsize*grid_point_v + grid_point_u].y, sum.y);
                }
                // Switch to new point
                sum = OutType(0.0, 0.0);
                grid_point_u = myGridU;
                grid_point_v = myGridV;
            }
            
            //TODO: Re-do the w-kernel/gcf for our data.
            Complex32 px = kernels[vi*maxsupport*maxsupport + myConvV * maxsupport + myConvU];// ??
            // Sum up
            InType temp = d_in[vi];
            OutType vi_v = OutType(temp.x, temp.y);
            sum = Complexfcma(px, vi_v, sum);
        }
        
        if( grid_point_u >= 0 && grid_point_u < gridsize && \
            grid_point_v >= 0 && grid_point_v < gridsize ) {
            atomicAdd(&d_out[grid_s + gridsize*grid_point_v + grid_point_u].x, sum.x);
            atomicAdd(&d_out[grid_s + gridsize*grid_point_v + grid_point_u].y, sum.y);
        }
    }
}

template<typename InType, typename OutType>
inline void launch_romein_kernel(int        nbaseline,
                                 int        maxsupport, 
                                 int        gridsize, 
                                 int        nbatch,
                                 int*       positions,
                                 Complex32* kernels,
                                 InType*    d_in,
                                 OutType*   d_out,
                                 cudaStream_t stream=0) {
    //cout << "LAUNCH for " << nelement << endl;
    // TODO: Is this really the best setup to use?
    dim3 block(8,1);
    dim3 grid(nbatch,1);
    /*
    cout << "  Block size is " << block.x << " by " << block.y << endl;
    cout << "  Grid  size is " << grid.x << " by " << grid.y << endl;
    */
    
    void* args[] = {&nbaseline,
                    &maxsupport,
                    &gridsize, 
                    &nbatch,
                    &positions,
                    &kernels,
                    &d_in,
                    &d_out};
    BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)romein_kernel<InType,OutType>,
                                             grid, block,
                                             &args[0], 0, stream),
                            BF_STATUS_INTERNAL_ERROR);
}

class BFromein_impl {
    typedef int    IType;
    typedef double FType;
public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
    typedef float  DType;
private:
    IType        _ntime;
    IType        _nchan;
    IType        _npol;
    IType        _nbaseline;
    IType        _maxsupport;
    IType        _gridsize;
    int*         _xyz = NULL;
    Complex32*   _kernels = NULL;
    cudaStream_t _stream;
public:
    BFromein_impl() : _ntime(1), _nchan(1), _npol(1), _nbaseline(1), \
                      _maxsupport(1), _stream(g_cuda_stream) {}
    inline IType ntime()      const { return _ntime;      }
    inline IType nchan()      const { return _nchan;      }
    inline IType npol()       const { return _npol;       }
    inline IType nbaseline()  const { return _nbaseline;  }
    inline IType maxsupport() const { return _maxsupport; }
    inline IType gridsize()   const { return _gridsize;   }
    void init(IType ntime, 
              IType nchan, 
              IType npol,
              IType nbaseline,
              IType maxsupport, 
              IType gridsize) {
        BF_TRACE();
        _ntime      = ntime;
        _nchan      = nchan;
        _npol       = npol;
        _nbaseline  = nbaseline;
        _maxsupport = maxsupport;
        _gridsize   = gridsize;
    }
    void set_positions(BFarray const* positions) { 
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        BF_ASSERT_EXCEPTION(positions->dtype == BF_DTYPE_I32, BF_STATUS_UNSUPPORTED_DTYPE);
        
        _xyz = (int*) positions->data;
    }
    void set_kernels(BFarray const* kernels) {
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        BF_ASSERT_EXCEPTION(kernels->dtype == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
        
        _kernels = (Complex32*) kernels->data;
    }
    void execute(BFarray const* in, BFarray const* out) {
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        BF_ASSERT_EXCEPTION(_xyz != NULL, BF_STATUS_INVALID_STATE);
        BF_ASSERT_EXCEPTION(_kernels != NULL, BF_STATUS_INVALID_STATE);
        BF_ASSERT_EXCEPTION(out->dtype == BF_DTYPE_CF32 \
                                          || BF_DTYPE_CF64, BF_STATUS_UNSUPPORTED_DTYPE);
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
        
        int nbatch = in->shape[0]*in->shape[1]*in->shape[2];
        
#define LAUNCH_ROMEIN_KERNEL(IterType,OterType) \
        launch_romein_kernel(_nbaseline, _maxsupport, _gridsize, nbatch, \
                             _xyz, _kernels, \
                             (IterType)in->data, (OterType)out->data, \
                             _stream)
        
        switch( in->dtype ) {
            case BF_DTYPE_CI8:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ROMEIN_KERNEL(char2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_ROMEIN_KERNEL(char2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                };
                break;
            case BF_DTYPE_CI16:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ROMEIN_KERNEL(short2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_ROMEIN_KERNEL(short2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ROMEIN_KERNEL(int2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_ROMEIN_KERNEL(int2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ROMEIN_KERNEL(long2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_ROMEIN_KERNEL(long2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ROMEIN_KERNEL(float2*, Complex32*);   break;
                    case BF_DTYPE_CF64: LAUNCH_ROMEIN_KERNEL(float2*, Complex64*);   break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ROMEIN_KERNEL(double2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_ROMEIN_KERNEL(double2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        }
#undef LAUNCH_ROMEIN_KERNEL
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
    }
    void set_stream(cudaStream_t stream) {
        _stream = stream;
    }
};

BFstatus bfRomeinCreate(BFromein* plan_ptr) {
    BF_TRACE();
    BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN_ELSE(*plan_ptr = new BFromein_impl(),
                       *plan_ptr = 0);
}

BFstatus bfRomeinInit(BFromein       plan,
                      BFarray const* positions,
                      BFarray const* kernels,
                      BFsize         gridsize) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(positions,                BF_STATUS_INVALID_POINTER);
    BF_ASSERT(positions->ndim == 5,     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(positions->shape[4] == 3, BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(space_accessible_from(positions->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_ASSERT(kernels,                                BF_STATUS_INVALID_POINTER);
    BF_ASSERT(kernels->ndim == 6,                     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(kernels->shape[4] == kernels->shape[5], BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(space_accessible_from(kernels->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    
    // Discover the dimensions of the positions/kernels.
    int ntime, nchan, npol, nbaseline, maxsupport;
    ntime = positions->shape[0];
    nchan = positions->shape[1];
    npol = positions->shape[2];
    nbaseline = positions->shape[3];
    maxsupport = kernels->shape[5];
    
    BF_TRY(plan->init(ntime, nchan, npol, nbaseline, maxsupport, gridsize));
    BF_TRY(plan->set_positions(positions));
    BF_TRY_RETURN(plan->set_kernels(kernels));
}
BFstatus bfRomeinSetStream(BFromein    plan,
                           void const* stream) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}
BFstatus bfRomeinSetPositions(BFromein       plan,
                              BFarray const* positions) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(positions,            BF_STATUS_INVALID_POINTER);
    BF_ASSERT(positions->ndim == 5, BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(positions->shape[0] == plan->ntime(),     BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(positions->shape[1] == plan->nchan(),     BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(positions->shape[2] == plan->npol(),      BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(positions->shape[3] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(positions->shape[4] == 3,                 BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(space_accessible_from(positions->space,   BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_TRY_RETURN(plan->set_positions(positions));
}
BFstatus bfRomeinSetKernels(BFromein       plan, 
                            BFarray const* kernels) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(kernels,            BF_STATUS_INVALID_POINTER);
    BF_ASSERT(kernels->ndim == 6, BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(kernels->shape[0] == plan->ntime(),      BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(kernels->shape[1] == plan->nchan(),      BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(kernels->shape[2] == plan->npol(),       BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(kernels->shape[3] == plan->nbaseline(),  BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(kernels->shape[4] == plan->maxsupport(), BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(kernels->shape[5] == plan->maxsupport(), BF_STATUS_INVALID_SHAPE  );
    
    BF_ASSERT(space_accessible_from(kernels->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_TRY_RETURN(plan->set_kernels(kernels));
}
BFstatus bfRomeinExecute(BFromein          plan,
                         BFarray const* in,
                         BFarray const* out) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
    BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
    BF_ASSERT( in->ndim == 4,          BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->ndim == in->ndim+1, BF_STATUS_INVALID_SHAPE);
    
//     TODO: What all should we support here?
//     BFarray in_flattened;
//     if( in->ndim > 4 ) {
//         // Keep the last three dim but attempt to flatten all others
//         unsigned long keep_dims_mask = padded_dims_mask(in);
//         keep_dims_mask |= 0x1 << (out->ndim-1);
//         keep_dims_mask |= 0x1 << (out->ndim-2);
//         keep_dims_mask |= 0x1 << (out->ndim-3);
//         keep_dims_mask |= 0x1 << (out->ndim-4);
//         flatten(in,   &in_flattened, keep_dims_mask);
//         in  =  &in_flattened;
//         BF_ASSERT(in_flattened.ndim == 4, BF_STATUS_UNSUPPORTED_SHAPE);
//     }
    BF_ASSERT( in->shape[0] == plan->ntime(),     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT( in->shape[1] == plan->nchan(),     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT( in->shape[2] == plan->npol(),      BF_STATUS_INVALID_SHAPE);
    BF_ASSERT( in->shape[3] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE);
    
//     TODO: What all should we support here?
//     BFarray out_flattened;
//     if( out->ndim > 5 ) {
//         // Keep the last four dim but attempt to flatten all others
//         unsigned long keep_dims_mask = padded_dims_mask(out);
//         keep_dims_mask |= 0x1 << (out->ndim-1);
//         keep_dims_mask |= 0x1 << (out->ndim-2);
//         keep_dims_mask |= 0x1 << (out->ndim-3);
//         keep_dims_mask |= 0x1 << (out->ndim-4);
//         keep_dims_mask |= 0x1 << (out->ndim-5);
//         flatten(out,   &out_flattened, keep_dims_mask);
//         out  =  &out_flattened;
//         BF_ASSERT(out_flattened.ndim == 5, BF_STATUS_UNSUPPORTED_SHAPE);
//     }
    BF_ASSERT(out->shape[0] == in->shape[0],     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[1] == plan->nchan(),    BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[2] == plan->npol(),     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[3] == plan->gridsize(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[4] == plan->gridsize(), BF_STATUS_INVALID_SHAPE);
    
    BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_TRY_RETURN(plan->execute(in, out));
}

BFstatus bfRomeinDestroy(BFromein plan) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    delete plan;
    return BF_STATUS_SUCCESS;
}




// Fix this to use templates properly.
BFstatus romein_float(BFarray const* data, // Our data, strided by d
		      BFarray const* uvgrid, // Our UV Grid to Convolve onto, strided by g
		      BFarray const* illum, // Our convolution kernel.
		      BFarray const* data_xloc,
		      BFarray const* data_yloc,
		      BFarray const* data_zloc,
		      int max_support,
		      int grid_size,
		      int data_size,
		      int nbatch){


    //TODO: I think remove these as the overhead is probably quite high...
    BF_TRACE();
    //BF_ASSERT(uvgrid && data && illum && data_xloc && data_yloc && data_zloc,
    //	      BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(uvgrid->dtype == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(data->dtype == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(illum->dtype == BF_DTYPE_CF32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(data_xloc->dtype == BF_DTYPE_I32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(data_yloc->dtype == BF_DTYPE_I32, BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(data_zloc->dtype == BF_DTYPE_I32, BF_STATUS_UNSUPPORTED_DTYPE);


    BF_ASSERT(space_accessible_from(data->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(uvgrid->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(illum->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(data_xloc->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(data_yloc->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(data_zloc->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
    
    void const* dptr = data->data;
    void const* uvgridptr = uvgrid->data;
    void const* illumptr = illum->data;
    void const* xloc = data_xloc->data;
    void const* yloc = data_yloc->data;
    void const* zloc = data_zloc->data;
    cuda::child_stream stream(g_cuda_stream);
    BF_TRACE_STREAM(stream);
    
    scatter_grid_kernel <<< nbatch, 8, 0, stream >>> ((cuComplex*)dptr,
						  (cuComplex*)uvgridptr,
						  (cuComplex*)illumptr,
						  (int*)xloc,
						  (int*)yloc,
						  (int*)zloc,
						  max_support,
						  grid_size,
						  data_size);
    
    //cudaError_t err = cudaGetLastError();
    //std::cout << "Error: " << cudaGetErrorString(err) << "\n";
    return BF_STATUS_SUCCESS;
}
		      
