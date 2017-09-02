
# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of The Bifrost Authors nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import absolute_import

from bifrost.pipeline import TransformBlock
from bifrost.linalg import LinAlg

from copy import deepcopy

class CorrelateBlock(TransformBlock):
    def __init__(self, iring, nframe_per_integration,
                 *args, **kwargs):
        super(CorrelateBlock, self).__init__(iring, *args, **kwargs)
        self.nframe_per_integration = nframe_per_integration
        self.linalg = LinAlg()
    def define_valid_input_spaces(self):
        return ('cuda',)
    def define_output_nframes(self, input_nframe):
        """Return output nframe, given input_nframes.
        """
        return 1
    def on_sequence(self, iseq):
        self.nframe_integrated = 0
        ihdr = iseq.header
        itensor = ihdr['_tensor']
        assert(itensor['labels'] == ['time', 'freq', 'station', 'pol'])
        ohdr = deepcopy(ihdr)
        otensor = ohdr['_tensor']
        otensor['dtype'] = 'cf32'
        for key in ['shape', 'labels', 'scales', 'units']:
            time_val, freq_val, stand_val, pol_val = itensor[key]
            otensor[key] = [time_val, freq_val,
                            stand_val, pol_val,
                            stand_val, pol_val]
        # Append subscripts to stand and pol axis labels
        for i in xrange(2):
            otensor['labels'][2+i] += '_i'
            otensor['labels'][4+i] += '_j'
        # Update time scale
        otensor['scales'][0][1] *= self.nframe_per_integration
        # Note: For future reference, possible values for this entry could be:
        #         full, hermitian, lower, upper, strictly_lower, strictly_upper
        ohdr['matrix_fill_mode'] = 'lower'
        ohdr['gulp_nframe'] = min(ohdr['gulp_nframe'],
                                  self.nframe_per_integration)
        # Note: User can override by setting self.gulp_nframe
        gulp_nframe_actual = self.gulp_nframe or ohdr['gulp_nframe']
        if self.nframe_per_integration % gulp_nframe_actual != 0:
            raise ValueError(
                "gulp_nframe (%i) does not divide " % gulp_nframe_actual +
                "nframe_per_integration (%i)" % self.nframe_per_integration)
        return ohdr
    def on_data(self, ispan, ospan):
        idata = ispan.data
        odata = ospan.data
        # TODO: Consider allowing returning (nframe_release, nframe_commit)
        #         from on_data, to enable flexible decoupling of
        #         the amount of data read/written during each gulp.
        beta = 0 if self.nframe_integrated == 0 else 1

        ntime, nchan, nstand, npol = idata.shape
        # Squash stand + pol axes together and permute to get the right matmul
        idata_mm = idata.reshape([ntime, nchan, nstand * npol]) \
                        .transpose([1, 0, 2])
        odata_mm = odata.reshape([nchan, nstand * npol, nstand * npol])
        # Check that the memory addresses haven't changed
        assert(idata_mm.ctypes.data == idata.ctypes.data)
        assert(odata_mm.ctypes.data == odata.ctypes.data)
        self.linalg.matmul(1, None, idata_mm, beta, odata_mm)

        self.nframe_integrated += ispan.nframe
        assert(self.nframe_integrated <= self.nframe_per_integration)
        if self.nframe_integrated == self.nframe_per_integration:
            self.nframe_integrated = 0
            return 1
        else:
            return 0

def correlate(iring, nframe_per_integration, *args, **kwargs):
    """Cross-multiply different stations and accumulate in time

    This is the X step of an FX correlator.

    Args:
        iring (Ring or Block): Input data source.
        nframe_per_integration (int): No. frames to integrate before
          producing an output frame of visibilities.
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.

    **Tensor semantics**::

        Input:  ['time', 'freq', 'station', 'pol'], dtype = any complex, space = CUDA
        Output: ['time', 'freq', 'station_i', 'pol_i', 'station_j', 'pol_j'] (lower triangle filled), dtype = cf32, space = CUDA

    Returns:
        CorrelateBlock: A new block instance.

    References:

        This block is backed by a fast GPU kernel based on the one in the xGPU
        library; see [2]_.

        .. [2] M. A. Clark, P. C. La Plante, and L. J. Greenhill,
               "Accelerating Radio Astronomy Cross-Correlation with Graphics
               Processing units", [arXiv:1107.4264 [astro-ph]].

        https://github.com/GPU-correlators/xGPU
    """
    return CorrelateBlock(iring, nframe_per_integration, *args, **kwargs)
