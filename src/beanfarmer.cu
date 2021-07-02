#include "cuda.hpp"
#include <bifrost/array.h>
#include <bifrost/beanfarmer.h>
#include <bifrost/common.h>
#include <bifrost/ring.h>
#include <cuComplex.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <utils.hpp>

#define WARP_SIZE 32
#define NTHREADS 1024
#define NWARPS_PER_BLOCK NTHREADS / WARP_SIZE

//#define DCP_DEBUG

extern "C" {

struct ComplexInt8 {
  int8_t x;
  int8_t y;
};

struct char4x2 {
  char4 x;
  char4 y;
};

struct char2x4 {
  char2 x;
  char2 y;
  char2 z;
  char2 w;
};

__forceinline__ __device__ void dp4a(int &c, const int &a, const int &b) {
#if __CUDA_ARCH__ >= 610
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "+r"(c) : "r"(a), "r"(b), "r"(c));
#else
  char4 &a4 = *((char4 *)&a);
  char4 &b4 = *((char4 *)&b);
  c += a4.x * b4.x;
  c += a4.y * b4.y;
  c += a4.z * b4.z;
  c += a4.w * b4.w;
#endif
}

__forceinline__ __device__ int2 int2_transpose(int2 const &input) {
  char2x4 a;
  char4x2 b;
  a = (*(char2x4 *)&input);
  b.x.x = a.x.x;
  b.x.y = a.y.x;
  b.x.z = a.z.x;
  b.x.w = a.w.x;
  b.y.x = a.x.y;
  b.y.y = a.y.y;
  b.y.z = a.z.y;
  b.y.w = a.w.y;
  return (*(int2 *)&b);
}

/**
 * @brief      Perform beamforming followed by detection and integration in
 * time.
 *
 * @param      aptf_voltages  Raw voltages in antenna, polarisation, time,
 * frequency order (fastest to slowest)
 * @param      apbf_weights   Beamforming weights in antenna, time, beam,
 * frequency order (fastest to slowest)
 * @param      tbf_powers     Output detected integrated powers in time, beam,
 * frequency order (fastest to slowest)
 * @param      NANTENNAS      Number of antennas in array
 * @param      NPOL           Number of polarizations (1 or 2)
 * @param      NCHANNELS      Number of frequency channels
 * @param      NACCUMULATE    Number of time samples over which to accumulate
 * after detection.
 */
__global__ void bf_aptf_general_k(int2 const *__restrict__ aptf_voltages,
                                  int2 const *__restrict__ apbf_weights,
                                  float *__restrict__ tbf_powers,
                                  int const NANTENNAS, int const NPOL,
                                  int const NCHANNELS, int const NBEAMS,
                                  int const NACCUMULATE, int const NSAMPLES) {
  int const NSAMPLES_PER_BLOCK = (NACCUMULATE * NTHREADS / WARP_SIZE);
  // int const NSAMPLES = (NSAMPLES_PER_BLOCK * 100);

  /**
   * Allocated shared memory to store beamforming weights and temporary space
   * for antenna data.
   */
  //__shared__ int2 shared_apb_weights[NANTENNAS/4][NPOL][WARP_SIZE];
  //__shared__ int2 shared_antennas[NTHREADS/WARP_SIZE][NANTENNAS/4];

  extern __shared__ int2 shm_array[];

  int2 *shared_apb_weights = (int2 *)shm_array;
  int2 *shared_antennas = (int2 *)&shm_array[NANTENNAS/4*NPOL*WARP_SIZE];
  // printf("SHM %d\n", shared_apb_weights[0][0][0].x);

  int const warp_idx = threadIdx.x / 0x20;
  int const lane_idx = threadIdx.x & 0x1f;

  /**
   * Each warp processes 32 beams (i.e. one beam per lane).
   */
  int const start_beam_idx = blockIdx.z * WARP_SIZE;

  /**
   * Complex multiply accumulators
   */
  int xx, yy, xy, yx;
  float power = 0.0f;
  int2 antennas, weights;
  int antenna_group_idx;
  int pol_stride = WARP_SIZE;
  int ant_stride = NPOL * WARP_SIZE;
  int sh_weights_idx = 0;
  int sh_ant_idx = 0;

  /**
   * Here we load all the beamforming weights neccessary for this block.
   * Implicit assumption here is that we do not need to change the weights over
   * the timescale of the data processed in one block. This is almost certainly
   * OK if the input data has already been rotated to telescope boresight and we
   * are only applying parallactic angle tracking updates.
   *
   * The global load is coalesced 8-byte (vectorised int2).
   */
  int const apbf_weights_offset =
      NANTENNAS / 4 * NPOL *
      (NBEAMS * blockIdx.y + (WARP_SIZE * blockIdx.z + warp_idx));

  for (int pol_idx = 0; pol_idx < NPOL; ++pol_idx) {
    for (antenna_group_idx = lane_idx; antenna_group_idx < NANTENNAS / 4;
         antenna_group_idx += WARP_SIZE) {
      sh_weights_idx =
          warp_idx + pol_idx * pol_stride + antenna_group_idx * ant_stride;
      shared_apb_weights[sh_weights_idx] = int2_transpose(
          apbf_weights[apbf_weights_offset + pol_idx * NANTENNAS / 4 +
                       antenna_group_idx]);
    }
  }
  // wait for all weights to load.
  __syncthreads();

  /**
   * Below is the main loop of the kernel. Here the kernel reads all the
   * antennas for a given sample and computes 32 beams. Each thread computes
   * only 1 beam and access to all the antennas required for that computation is
   * achieved via a shared memory broadcasts.
   */
  int sample_offset = NACCUMULATE * (blockIdx.x * NWARPS_PER_BLOCK + warp_idx);
  for (int sample_idx = sample_offset;
       sample_idx < (sample_offset + NACCUMULATE); ++sample_idx) {
    int aptf_voltages_partial_idx =
        NANTENNAS / 4 * NPOL * (NSAMPLES * blockIdx.y + sample_idx);
    for (int pol_idx = 0; pol_idx < NPOL; ++pol_idx) {
      // Set the complex accumulator to zero before adding the next polarisation
      xx = 0;
      yy = 0;
      xy = 0;
      yx = 0;

      /**
       * Load all antennas antennas required for this sample into shared memory.
       * Without an outer loop to allow for more antennas (which would also
       * require more shared memory), this kernel is limited to a max of 32 * 4
       * = 128 antennas in a sub-array.
       */
      if (lane_idx < NANTENNAS / 4) {
        sh_ant_idx = warp_idx * (NANTENNAS / 4) + lane_idx;
        shared_antennas[sh_ant_idx] =
            int2_transpose(aptf_voltages[aptf_voltages_partial_idx + lane_idx +
                                         NANTENNAS / 4 * pol_idx]);
      }

      /*Required to synchronise across all the blocks*/
      __threadfence_block();

      for (antenna_group_idx = 0; antenna_group_idx < NANTENNAS / 4;
           ++antenna_group_idx) {

        sh_ant_idx = warp_idx * (NANTENNAS / 4) + antenna_group_idx;
        antennas = shared_antennas[sh_ant_idx];

        // load corresponding 4 weights
        sh_weights_idx =
            lane_idx + pol_idx * pol_stride + antenna_group_idx * ant_stride;
        weights = shared_apb_weights[sh_weights_idx];

        // dp4a multiply add
        dp4a(xx, weights.x, antennas.x);
        dp4a(yy, weights.y, antennas.y);
        dp4a(xy, weights.x, antennas.y);
        dp4a(yx, weights.y, antennas.x);
        //if (threadIdx.x == 0 and threadIdx.y == 0 and blockIdx.x == 0 and blockIdx.y == 0) printf("W: %d A: %d\n", weights.x, antennas.x);
      }
      int r = xx - yy;
      int i = xy + yx;
      // be careful of overflow
      power += (float)(r * r + i * i);
    }
  }

  /**
   * As we have looped over both polarisation and sample in the above loop we
   * are now free to simply write back to global memory. Here we write back
   * uncoalesced to get the data in time beam order. The performance penalty
   * here is very small compared to the compute time in the rest of the kernel
   * as the total volume of data being written out is a factor of NACCUMULATE *
   * NANTENNAS / WARP_SIZE smaller than the input (e.g. for 64 antennas and 16
   * integrated samples this is a factor of 32).
   */
  int output_idx = (NWARPS_PER_BLOCK * gridDim.x) *
                       (NBEAMS * blockIdx.y + (start_beam_idx + lane_idx)) +
                   sample_offset / NACCUMULATE;
  tbf_powers[output_idx] = power;
}

void launch_beanfarmer(int2 const *__restrict__ aptf_voltages,
                       int2 const *__restrict__ apbf_weights,
                       float *__restrict__ tbf_powers, const int NANTENNAS,
                       const int NPOL, const int NCHANNELS, const int NBEAMS,
                       const int NACCUMULATE, const int NSAMPLES) {

  const int NSAMPLES_PER_BLOCK = (NACCUMULATE * NTHREADS / WARP_SIZE);
  //const int NSAMPLES = (NSAMPLES_PER_BLOCK * 100);
  int shm_bytes = sizeof(int2) * (NANTENNAS / 4 * NPOL * WARP_SIZE +
                                  NTHREADS / WARP_SIZE * NANTENNAS / 4);

  dim3 grid(NSAMPLES / (NWARPS_PER_BLOCK * NACCUMULATE), NCHANNELS,
            NBEAMS / WARP_SIZE);
  dim3 block(NTHREADS, 1, 1);

#ifdef DCP_DEBUG
  printf("Debug: NANT %d NPOL %d NBEAM %d NCHAN %d NACC %d NSAMP %d\n", NANTENNAS, NPOL,
         NBEAMS, NCHANNELS, NACCUMULATE, NSAMPLES);
  printf("Debug: <<<B: (%d, %d, %d) G: (%d, %d, %d) SHM: %dB >>>\n", block.x,
         block.y, block.z, grid.x, grid.y, grid.z, shm_bytes);
#endif
  // cudaStream_t stream = 0;

  bf_aptf_general_k<<<grid, NTHREADS, shm_bytes>>>(
      (int2 *)aptf_voltages, (int2 *)apbf_weights, (float *)tbf_powers,
      NANTENNAS, NPOL, NCHANNELS, NBEAMS, NACCUMULATE, NSAMPLES);
  cudaDeviceSynchronize();
}

BFstatus BeanFarmer(BFarray *voltages, BFarray *weights,
                    BFarray *beamformed_out, const int NACCUMULATE) {
  int2 *aptf_voltages = (int2 *)voltages->data;
  int2 *apbf_weights = (int2 *)weights->data;
  float *tbf_powers = (float *)beamformed_out->data;

  const int NSAMPLES  = voltages->shape[2];
  const int NANTENNAS = weights->shape[3];
  const int NPOL = weights->shape[2];
  const int NBEAMS = weights->shape[1];
  const int NCHANNELS = weights->shape[0];


  launch_beanfarmer(aptf_voltages, apbf_weights, tbf_powers, NANTENNAS, NPOL,
                    NCHANNELS, NBEAMS, NACCUMULATE, NSAMPLES);

  BF_CHECK_CUDA(cudaGetLastError(), BF_STATUS_DEVICE_ERROR);
  return BF_STATUS_SUCCESS;
}

} //ExternC
