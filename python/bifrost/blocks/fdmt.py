
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

# TODO: Add support for batching once it's supported by the backend

from __future__ import absolute_import

from bifrost.pipeline import TransformBlock
from bifrost.fdmt import Fdmt
from bifrost.units import convert_units

from copy import deepcopy
import math

class FdmtBlock(TransformBlock):
    def __init__(self, iring, max_dm,
                 exponent=-2.0, negative_delays=False,
                 *args, **kwargs):
        super(FdmtBlock, self).__init__(iring, *args, **kwargs)
        self.space    = self.orings[0].space
        self.max_dm   = max_dm
        self.kdm      = 4.148741601e3 # MHz**2 cm**3 s / pc
        self.dm_units = 'pc cm^-3'
        self.exponent = exponent
        self.negative_delays = negative_delays
        self.fdmt     = Fdmt()
    def define_valid_input_spaces(self):
        return ('cuda',)
    def on_sequence(self, iseq):
        ihdr = iseq.header
        itensor = ihdr['_tensor']
        # TODO: Assert that axis labels match expected (and/or allow more flexibility in which axes are used)
        nchan    = itensor['shape' ][-2]
        npol     = itensor['shape' ][-3]
        f0_, df_ = itensor['scales'][-2]
        t0_, dt_ = itensor['scales'][-1]
        # Units must match self.kdm
        f0 = convert_units(f0_, itensor['units'][-2], 'MHz')
        df = convert_units(df_, itensor['units'][-2], 'MHz')
        dt = convert_units(dt_, itensor['units'][-1], 's')
        rel_delay = (self.kdm / dt * self.max_dm *
                     (f0**-2 - (f0 + nchan * df)**-2))
        self.max_delay = int(math.ceil(abs(rel_delay)))
        self.dm_step   = self.max_dm / self.max_delay
        if self.negative_delays:
            self.dm_step *= -1
        self.fdmt.init(nchan, self.max_delay, f0, df, self.exponent, self.space)
        ohdr = deepcopy(ihdr)
        if 'refdm' in ihdr:
            refdm = convert_units(ihdr['refdm'], ihdr['refdm_units'], self.dm_units)
        else:
            refdm = 0.
        # Update transformed axis info
        ohdr['_tensor']['dtype']      = 'f32'
        ohdr['_tensor']['shape'][-2]  = self.max_delay
        ohdr['_tensor']['labels'][-2] = 'dispersion'
        ohdr['_tensor']['scales'][-2] = (refdm, self.dm_step)
        ohdr['_tensor']['units'][-2]  = self.dm_units
        # Add some new metadata
        ohdr['max_dm']       = self.max_dm
        ohdr['max_dm_units'] = self.dm_units
        ohdr['cfreq']        = 0.5 * (f0_ + (nchan - 1) * df_)
        ohdr['cfreq_units']  = itensor['units'][-2]
        ohdr['bw']           = nchan * df_
        ohdr['bw_units']     = itensor['units'][-2]
        gulp_nframe = self.gulp_nframe or ihdr['gulp_nframe']
        return ohdr, slice(0, gulp_nframe + self.max_delay, gulp_nframe)
    def on_data(self, ispan, ospan):
        if ispan.nframe <= self.max_delay:
            # Cannot fully process any frames
            return 0

        size = self.fdmt.get_workspace_size(ispan.data, ospan.data)
        with self.get_temp_storage(self.space).allocate(size) as temp_storage:
            self.fdmt.execute_workspace(ispan.data, ospan.data,
                                        temp_storage.ptr, temp_storage.size,
                                        negative_delays=self.negative_delays)
        return ispan.nframe - self.max_delay
        # ***TODO: Need to tell downstream blocks the *stride*, not the
        #            reserve size, because the stride is what determines
        #            how much output this block generates each gulp.
        #            HOWEVER, still can't fuse on the input side without
        #              deadlocking. Not sure if there's any way around this.

def fdmt(iring, max_dm, exponent=-2.0, negative_delays=False,
         *args, **kwargs):
    """Apply the Fast Dispersion Measure Transform (FDMT).

    This uses the GPU. It is used in pulsar and fast radio burst (FRB)
    search pipelines for dedispersing frequency data.

    Args:
        iring (Ring or Block): Input data source.
        max_dm (float): Max dispersion measure to search up to
            (in units of pc/cm^3).
        exponent (float): Frequency power law to search
            (-2.0 for interstellar dispersion).
        negative_delays (bool): If True, the transform applies dispersions
            in the range (-max_dm, 0] instead of [0, max_dm).
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.

    **Tensor semantics**::

        Input:  ['pol', 'freq',       'time'], dtype = any real, space = CUDA
        Output: ['pol', 'dispersion', 'time'], dtype = f32, space = CUDA

    Returns:
        FdmtBlock: A new block instance.

    References:

        This is based on Barak Zackay's FDMT algorithm, see [1]_.

        .. [1] Zackay, Barak, and Eran O. Ofek. "An accurate
           and efficient algorithm for detection of radio bursts
           with an unknown dispersion measure, for single dish
           telescopes and interferometers." arXiv preprint arXiv:1411.5373
           (2014).

    Note:
        The number of dispersion measure trials (delays) computed by
        this algorithm depends on the value of ``max_dm`` and the time
        and frequency scales of the input data. The
        ``bifrost.blocks.print_header`` block can be used to check the
        output dimensions if needed.

    """
    return FdmtBlock(iring, max_dm, exponent, negative_delays,
                     *args, **kwargs)
