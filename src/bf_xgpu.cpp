#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>

#include <bifrost/array.h>
#include <bifrost/common.h>
#include <utils.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <xgpu.h>

extern "C" {

static XGPUContext context;
static XGPUInfo info;

/* 
 * Initialize the xGPU library by providing
 * a pointer to the input and output data (on the host),
 * and a GPU device ID
 */
BFstatus bfXgpuInitialize(BFarray *in, BFarray *out, int gpu_dev) {
  int xgpu_error;
  xgpuInfo(&info);
  // Don't bother checking sizes if the input space is CUDA.
  // We're not going to use these arrays anyway
  if (in->space != BF_SPACE_CUDA) {
      if (num_contiguous_elements(in) != info.vecLength) {
        fprintf(stderr, "ERROR: xgpuInitialize: number of elements in != vecLength\n");
        fprintf(stderr, "number of elements in: %lu\n", num_contiguous_elements(in));
        fprintf(stderr, "vecLength: %llu\n", info.vecLength);
        return BF_STATUS_INVALID_SHAPE;
      }
      if (num_contiguous_elements(out) != info.matLength) {
        fprintf(stderr, "ERROR: xgpuInitialize: number of elements out != matLength\n");
        fprintf(stderr, "number of elements out: %lu\n", num_contiguous_elements(out));
        fprintf(stderr, "matLength: %llu\n", info.matLength);
        return BF_STATUS_INVALID_SHAPE;
      }
  }
  context.array_h = (SwizzleInput *)in->data;
  context.array_len = info.vecLength;
  context.matrix_h = (Complex *)out->data;
  context.matrix_len = info.matLength;
  if (in->space == BF_SPACE_CUDA) {
      xgpu_error = xgpuInit(&context, gpu_dev | XGPU_DONT_REGISTER | XGPU_DONT_MALLOC_GPU);
  } else {
      xgpu_error = xgpuInit(&context, gpu_dev);
  }
  if (xgpu_error != XGPU_OK) {
    fprintf(stderr, "ERROR: xgpuInitialize: call returned %d\n", xgpu_error);
    return BF_STATUS_INTERNAL_ERROR;
  } else {
    return BF_STATUS_SUCCESS;
  }
}

/*
 * Call the xGPU kernel.
 * in : pointer to input data array on host
 * out: pointer to output data array on host
 * doDump : if 1, this is the last call in an integration, and results
 *          will be copied to the host.
 */
BFstatus bfXgpuCorrelate(BFarray *in, BFarray *out, int doDump) {
  if (in->space == BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  if (out->space == BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  int xgpu_error;
  context.array_h = (SwizzleInput *)in->data;
  context.array_len = info.vecLength;
  context.matrix_h = (Complex *)out->data;
  context.matrix_len = info.matLength;
  xgpu_error = xgpuCudaXengineSwizzle(&context, doDump ? SYNCOP_DUMP : SYNCOP_SYNC_TRANSFER);
  if (doDump) {
    xgpuClearDeviceIntegrationBuffer(&context);
  }
  if (xgpu_error != XGPU_OK) {
    return BF_STATUS_INTERNAL_ERROR;
  } else {
    return BF_STATUS_SUCCESS;
  }
}

/*
 * Call the xGPU kernel having pre-copied data to device memory.
 * Note that this means xGPU can't take advantage of its inbuild
 * copy/compute pipelining.
 * in : pointer to input data array on device
 * out: pointer to output data array on device
 * doDump : if 1, this is the last call in an integration, and results
 *          will be copied to the host.
 */
static int newAcc = 1; // flush vacc on the first call
BFstatus bfXgpuKernel(BFarray *in, BFarray *out, int doDump) {
  if (in->space != BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  if (out->space != BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  int xgpu_error;
  context.array_h = (ComplexInput *)in->data;
  context.array_len = info.vecLength;
  context.matrix_h = (Complex *)out->data;
  context.matrix_len = info.matLength;
  xgpu_error = xgpuCudaXengineSwizzleKernel(&context, doDump ? SYNCOP_DUMP : 0, newAcc,
		                           (SwizzleInput *)in->data, (Complex *)out->data);

  if (newAcc) {
    newAcc = 0;
  }
  if (doDump) {
    newAcc = 1;
  }
  if (xgpu_error != XGPU_OK) {
    fprintf(stderr, "ERROR: xgpuKernel: kernel call returned %d\n", xgpu_error);
    return BF_STATUS_INTERNAL_ERROR;
  } else {
    return BF_STATUS_SUCCESS;
  }
}

/*
 * Given an xGPU accumulation buffer, grab a subset of visibilities from
 * and gather them in a new buffer, in order chan x visibility x complexity [int32]
 * BFarray *in : Pointer to a BFarray with storage in device memory, where xGPU results reside
 * BFarray *in : Pointer to a BFarray with storage in device memory where collated visibilities should be written.
 * BFarray *vismap : array of visibilities in [[polA, polB], [polC, polD], ... ] form.
 * int nchan_sum: The number of frequency channels to sum over
 */
BFstatus bfXgpuSubSelect(BFarray *in, BFarray *out, BFarray *vismap, int nchan_sum) {
  long long unsigned nvis = num_contiguous_elements(vismap);
  int xgpu_error;
  if (in->space != BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  if (out->space != BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  if (vismap->space != BF_SPACE_CUDA) {
    return BF_STATUS_UNSUPPORTED_SPACE;
  }
  xgpu_error = xgpuCudaSubSelect(&context, (Complex *)in->data, (Complex *)out->data, (int *)vismap->data, nvis, nchan_sum);
  if (xgpu_error != XGPU_OK) {
    fprintf(stderr, "ERROR: xgpuKernel: kernel call returned %d\n", xgpu_error);
    return BF_STATUS_INTERNAL_ERROR;
  } else {
    return BF_STATUS_SUCCESS;
  }
}

/* Computes the triangular index of an (i,j) pair as shown here...
 * NB: Output is valid only if i >= j.
 *
 *      i=0  1  2  3  4..
 *     +---------------
 * j=0 | 00 01 03 06 10
 *   1 |    02 04 07 11
 *   2 |       05 08 12
 *   3 |          09 13
 *   4 |             14
 *   :
 */
int tri_index(int i, int j){
  return (i * (i+1))/2 + j;
  }

/* Returns index into the GPU's register tile ordered output buffer for the
 * real component of the cross product of inputs in0 and in1.  Note that in0
 * and in1 are input indexes (i.e. 0 based) and often represent antenna and
 * polarization by passing (2*ant_idx+pol_idx) as the input number (NB: ant_idx
 * and pol_idx are also 0 based).  Return value is valid if in1 >= in0.  The
 * corresponding imaginary component is located xgpu_info.matLength words after
 * the real component.
 */
int regtile_index(int in0, int in1, int nstand) {
  int a0, a1, p0, p1;
  int num_words_per_cell=4;
  int quadrant, quadrant_index, quadrant_size, cell_index, pol_offset, index;
  a0 = in0 >> 1;
  a1 = in1 >> 1;
  p0 = in0 & 1;
  p1 = in1 & 1;
  
  // Index within a quadrant
  quadrant_index = tri_index(a1/2, a0/2);
  // Quadrant for this input pair
  quadrant = 2*(a0&1) + (a1&1);
  // Size of quadrant
  quadrant_size = (nstand/2 + 1) * nstand/4;
  // Index of cell (in units of cells)
  cell_index = quadrant*quadrant_size + quadrant_index;
  // Pol offset
  pol_offset = 2*p1 + p0;
  // Word index (in units of words (i.e. floats) of real component
  index = (cell_index * num_words_per_cell) + pol_offset;
  return index;
  }

BFstatus bfXgpuGetOrder(BFarray *antpol_to_input, BFarray *antpol_to_bl, BFarray *is_conj) {
  int *ip_map = (int *)antpol_to_input->data; // indexed by stand, pol
  int *bl_map = (int *)antpol_to_bl->data;    // indexed by stand0, stand1, pol0, pol1
  int *conj_map = (int *)is_conj->data;       // indexed by stand0, stand1, pol0, pol1
  int s0, s1, p0, p1, i0, i1;
  int nstand, npol;
  XGPUInfo xgpu_info;
  xgpuInfo(&xgpu_info);
  nstand = xgpu_info.nstation;
  npol = xgpu_info.npol;
  for (s0=0; s0<nstand; s0++) {
    for (s1=0; s1<nstand; s1++) {
      for (p0=0; p0<npol; p0++) {
        for (p1=0; p1<npol; p1++) {
          i0 = ip_map[npol*s0 + p0];
          i1 = ip_map[npol*s1 + p1];
          if (i1 >= i0) {
            bl_map[s0*nstand*npol*npol + s1*npol*npol + p0*npol + p1] = regtile_index(i0, i1, nstand);
            conj_map[s0*nstand*npol*npol + s1*npol*npol + p0*npol + p1] = 0;
          } else {
            bl_map[s0*nstand*npol*npol + s1*npol*npol + p0*npol + p1] = regtile_index(i1, i0, nstand);
            conj_map[s0*nstand*npol*npol + s1*npol*npol + p0*npol + p1] = 1;
          }
        }
      }
    }
  }
  return BF_STATUS_SUCCESS;
}

/*
 * Reorder a DP4A xGPU spec output into something more sane, throwing
 * away unwanted baselines and re-concatenating real and imag parts in
 * a reasonable way.
 */
BFstatus bfXgpuReorder(BFarray *xgpu_output, BFarray *reordered, BFarray *baselines, BFarray *is_conjugated) {
  XGPUInfo xgpu_info;
  xgpuInfo(&xgpu_info);

  int *output = (int *)reordered->data;
  int *input_r = (int *)xgpu_output->data;
  int *input_i = input_r + xgpu_info.matLength;
  int *bl = (int *)baselines->data;
  int *conj = (int *)is_conjugated->data;
  int n_bl = num_contiguous_elements(baselines);
  int xgpu_n_input = xgpu_info.nstation * xgpu_info.npol;
  int n_chan = xgpu_info.nfrequency;
  int i, c;
  // number of entries per channel
  size_t regtile_chan_len = 4 * 4 * xgpu_n_input/4 * (xgpu_n_input/4+1) / 2;
  fprintf(stderr, "nbaselines: %d; nchans:%d\n", n_bl, n_chan);
  for (i=0; i<n_bl; i++) {
    for (c=0; c<n_chan; c++) {
      output[2*i*n_chan + 2*c]     = input_r[c*regtile_chan_len + bl[i]];
      if ( conj[i] ) {
        output[2*i*n_chan + 2*c + 1] = input_i[c*regtile_chan_len + bl[i]];
      } else {
        output[2*i*n_chan + 2*c + 1] = -input_i[c*regtile_chan_len + bl[i]];
      }
    }
  }
  return BF_STATUS_SUCCESS;
}

} // C
