#include <bifrost/fft_shift.h>
#include <cuComplex.h>
#include "assert.hpp"
#include "trace.hpp"
#include "utils.hpp"
#include "cuda.hpp"
#include "cuda/stream.hpp"

// This explicily fft shifts a 2D space. 

template <typename T>
__global__ void fft_shift_kernel_2d_k(T *grid, int size){

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x<size/2 && y <size){

    int ix0 = y * size + x;
    int ix1 = (ix0 + (size + 1) * (size/2)) % (size*size);

    T temp = grid[ix0];
    grid[ix0] = grid[ix1];
    grid[ix1] = temp;
  }
}


template <typename T>
__host__ BFstatus fft_shift_2di(BFarray const *grid, int size, int batch_no){

    void const * dptr = grid->data;
    
    int gs = std::min(size, 32);
    int bs = std::max(size/gs, 1);
    dim3 dimGrid(bs,bs);
    dim3 dimBlock(gs,gs);
    cuda::child_stream stream(g_cuda_stream);
    BF_TRACE_STREAM(stream);

    for(int i = 0; i<batch_no; ++i){
	int offset = i * size * size;
	fft_shift_kernel_2d_k <<< dimGrid, dimBlock, 0, stream >>> ((T*)dptr + offset, size);
    }	
    return BF_STATUS_SUCCESS;

}


__host__ BFstatus fft_shift_2d(BFarray const *grid, int size, int batch_no){
    // Assume square matrix for simplicity...
    BF_TRACE();
    
    BF_ASSERT(grid,                         BF_STATUS_INVALID_POINTER);
    BF_ASSERT(grid->ndim >= 3,              BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(grid->shape[grid->ndim-2] \
              == grid->shape[grid->ndim-1], BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(grid->shape[grid->ndim-1] \
              == size,                      BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(space_accessible_from(grid->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
    
    // Check leading dimensions
    int nb = 1;
    for(int i=0; i<grid->ndim-2; ++i) {
        nb *= grid->shape[i];
    }
    BF_ASSERT(nb == batch_no, BF_STATUS_INVALID_SHAPE);

    switch(grid->dtype) {
    case BF_DTYPE_I8:   return fft_shift_2di<int8_t>(grid, size, batch_no);
    case BF_DTYPE_I16:  return fft_shift_2di<int16_t>(grid, size, batch_no);
    case BF_DTYPE_U8:   return fft_shift_2di<uint8_t>(grid, size, batch_no);
    case BF_DTYPE_U16:  return fft_shift_2di<uint16_t>(grid, size, batch_no);
    case BF_DTYPE_F32:  return fft_shift_2di<float>(grid, size, batch_no);
    case BF_DTYPE_F64:  return fft_shift_2di<double>(grid, size, batch_no);
    case BF_DTYPE_CF32: return fft_shift_2di<cuComplex>(grid, size, batch_no);
    case BF_DTYPE_CF64: return fft_shift_2di<cuDoubleComplex>(grid, size, batch_no);
    default: BF_FAIL("Unsupported dtype", BF_STATUS_UNSUPPORTED_DTYPE);
    }
}
