
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
from bifrost.fdmt import Fdmt
from bifrost.units import convert_units

from copy import deepcopy
import math

class FdmtBlock(TransformBlock):
    def __init__(self, iring, max_dm=None, max_delay=None, max_diagonal=None,
                 exponent=-2.0, negative_delays=False,
                 *args, **kwargs):
        super(FdmtBlock, self).__init__(iring, *args, **kwargs)
        if sum([m is not None for m in [max_dm, max_delay, max_diagonal]]) != 1:
            raise ValueError("Must specify exactly one of: max_dm, max_delay, "
                             "max_diagonal")
        self.space     = self.orings[0].space
        self.max_value = max_dm or max_delay or max_diagonal or 0.
        self.max_mode  = ('dm' if max_dm is not None else
                          'delay' if max_delay is not None else
                          'diagonal' if max_diagonal is not None else
                          'error')
        self.kdm       = 4.148741601e3 # MHz**2 cm**3 s / pc
        self.dm_units  = 'pc cm^-3'
        self.exponent  = exponent
        self.negative_delays = negative_delays
        self.fdmt      = Fdmt()
    def define_valid_input_spaces(self):
        return ('cuda',)
    def on_sequence(self, iseq):
        ihdr = iseq.header
        itensor = ihdr['_tensor']
        labels = itensor['labels']
        if labels[-1] != 'time' or labels[-2] != 'freq':
            raise KeyError("Expected axes [..., 'freq', 'time'], got %s"
                           % labels)
        nchan    = itensor['shape' ][-2]
        f0_, df_ = itensor['scales'][-2]
        t0_, dt_ = itensor['scales'][-1]
        # Units must match self.kdm
        f0 = convert_units(f0_, itensor['units'][-2], 'MHz')
        df = convert_units(df_, itensor['units'][-2], 'MHz')
        dt = convert_units(dt_, itensor['units'][-1], 's')
        if self.max_mode == 'diagonal':
            max_diagonal = self.max_value
            self.max_mode = 'delay'
            self.max_value = int(math.ceil(nchan * max_diagonal))
        if self.max_mode == 'dm':
            max_dm = self.max_value
            rel_delay = (self.kdm / dt * max_dm *
                         (f0**-2 - (f0 + nchan * df)**-2))
            self.max_delay = int(math.ceil(abs(rel_delay)))
        elif self.max_mode == 'delay':
            self.max_delay = self.max_value
            fac = (f0**-2 - (f0 + nchan * df)**-2)
            max_dm = self.max_delay * dt / (self.kdm * abs(fac))
        else:
            raise ValueError("Unknown max mode: %s" % self.max_mode)
        if self.negative_delays:
            max_dm = -max_dm
        self.dm_step = max_dm / self.max_delay
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
        ohdr['max_dm']       = max_dm
        ohdr['max_dm_units'] = self.dm_units
        ohdr['cfreq']        = f0_ + 0.5 * (nchan - 1) * df_
        ohdr['cfreq_units']  = itensor['units'][-2]
        ohdr['bw']           = nchan * df_
        ohdr['bw_units']     = itensor['units'][-2]
        gulp_nframe = self.gulp_nframe or ihdr['gulp_nframe']
        return ohdr
    def define_input_overlap_nframe(self, iseq):
        """Return no. input frames that should overlap between successive spans.
        """
        return self.max_delay
    def on_data(self, ispan, ospan):
        if ispan.nframe <= self.max_delay:
            # Cannot fully process any frames
            return 0
        size = self.fdmt.get_workspace_size(ispan.data, ospan.data)
        with self.get_temp_storage(self.space).allocate(size) as temp_storage:
            self.fdmt.execute_workspace(ispan.data, ospan.data,
                                        temp_storage.ptr, temp_storage.size,
                                        negative_delays=self.negative_delays)

def fdmt(iring, max_dm=None, max_delay=None, max_diagonal=None,
         exponent=-2.0, negative_delays=False,
         *args, **kwargs):
    """Apply the Fast Dispersion Measure Transform (FDMT).

    This uses the GPU. It is used in pulsar and fast radio burst (FRB)
    search pipelines for dedispersing channelised data.

    Args:
        iring (Ring or Block): Input data source.
        max_dm (float): Max dispersion measure to search up to
            (in units of pc/cm^3).
        max_delay (int): Max dispersion delay across the band to search
            up to (in units of time samples).
        max_diagonal (float): Max dispersion delay across the band to
            search up to (in units of the number of frequency channels).
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
        Only one of max_dm, max_delay, or max_diagonal may be specified.
        In the case of ``max_dm``, the number of dispersion measure trials
        (delays) computed by this algorithm depends on the value of ``max_dm``
        and the time and frequency scales of the input data. The
        ``bifrost.blocks.print_header`` block can be used to check the
        output dimensions if needed.

    """
    return FdmtBlock(iring, max_dm, max_delay, max_diagonal,
                     exponent, negative_delays,
                     *args, **kwargs)
