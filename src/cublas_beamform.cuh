#ifndef _CUBLAS_BEAMFORM_H
#define _CUBLAS_BEAMFORM_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Transpose time x chan x pol x 4+4 bit to
#define TRANSPOSE_POL_BLOCK_SIZE 8
// chan x pol x time x 32+32 bit float
__global__ void trans_4bit_to_float(unsigned char *in,
                                    float *out,
                                    int n_pol,
                                    int n_chan,
                                    int n_time
                                   );

// Transpose chan x beam x pol x time x 32+32 float to
// beam x time[part-summed] x chan x [XX,YY,XY*_r,XY*_i] x 32 float
// Each thread deals with two pols of a beam, and sums over n_time_sum time samples
__global__ void trans_output_and_sum(float *in,
                                    float *out,
                                    int n_chan,
                                    int n_beam,
                                    int n_time,
                                    int n_time_sum
                                   );

__global__ void complex2pow(float *in, float *out, int N);

void cublas_beamform_destroy();
void cublas_beamform(unsigned char *in4_d, float *sum_out_d, float *weights_d);
void cublas_beamform_init(int device, int ninputs, int nchans, int ntimes, int nbeams, int ntimeblocks);

#endif
