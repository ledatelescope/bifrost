#include <bifrost/common.h>
#include <bifrost/array.h>

/*
 * gpudev: GPU device ID to use
 * ninputs: Number of inputs (single-polarization) to the beamformer
 * nchans: Number of frequency channels
 * ntimes: Number of time samples per beamforming call
 * nbeams: Number of beams to generate. If using ntime_blocks > 0, beams=N will deliver N/2 beams.
 *         (See bfBeamformRun)
 * ntime_blocks: Number of time blocks to output. Eg. if ntimes=1000 and ntime_blocks=10, the beamformer
                 will integrate over 100 samples per call. Set to 0 for no accumulation, in which case
                 raw beam voltages are output.
 */
BFstatus bfBeamformInitialize(
  int gpudev,
  int ninputs,
  int nchans,
  int ntimes,
  int nbeams,
  int ntime_blocks
);

/*
 * in:  Pointer to ntime x nchan x ninputs x 4+4 bit data block
 * out: Pointer to output data.
 *        If ntime_blocks > 0: !!!!UNTESTED, probably broken!!!!
 *          For the purposes of generating dynamic spectra, beam 2n and 2n+1 are considered
 *          to be two pols of the same pointing, and are cross-multipled and summed over
 *          ntimes/ntime_blocks to form the output array:
 *            nbeam/2 x ntime_blocks x nchan x 4 x float32 (powers, XX, YY, re(XY, im(XY))
 *          Note that this means for N dual-pol beam pointings, the beamformer should be
 *          constructed with nbeams=2N. This isn't very efficient, but makes it easy to deal
 *          with arbitrary polarization orderings in the input buffer (suitable beamforming
 *          coefficients can make appropriate single-pol beam pairs).
 *        If ntime_blocks = 0:
 *          Data are returned as voltages, in order:
 *            nchan x nbeam x ntime x complex64 beamformer block
 *                            
 * weights -- pointer to nbeams x nchans x ninputs x complex64 weights
 */
BFstatus bfBeamformRun(
  BFarray *in,
  BFarray *out,
  BFarray *weights
);

/*
 * Take the output of bfBeamformRun with ntime_blocks = 0, and perform transposing and integration
 * of data, to deliver a time integrated dual-pol dynamic spectra of the form:
 *   nbeam/2 x ntime/ntimes_sum x nchan x 4 x float32 (powers, XX, YY, re(XY, im(XY))
 * I.e., the format which would be returned by bfBeamformRun if ntime_blocks > 0
 */
BFstatus bfBeamformIntegrate(
  BFarray *in,
  BFarray *out,
  int ntimes_sum
);

/*
 * Take the output of bfBeamformRun with ntime_blocks = 0, and 
 * deliver a time integrated dual-pol dynamic spectra for a single beam of the form:
 *   ntime/ntimes_sum x nchan x 4 x float32 (powers, XX, YY, re(XY, im(XY))
 * 
 * ntime_sum: the number of times to integrate
 * beam_index: The beam to select (if beam_index=N, beams N and N+1 will be used as a polarization pair)
 */
BFstatus bfBeamformIntegrateSingleBeam(
  BFarray *in,
  BFarray *out,
  int ntimes_sum,
  int beam_index
);
