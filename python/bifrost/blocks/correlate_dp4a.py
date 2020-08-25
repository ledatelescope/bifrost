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

import bifrost as bf
from bifrost.pipeline import TransformBlock
from bifrost.DataType import DataType
from bifrost.libbifrost import _bf
from datetime import datetime
import numpy as np

from copy import deepcopy
        
class CorrelateDp4aBlock(bf.pipeline.TransformBlock):
    def __init__(self, iring, nframe_to_avg=1,
                 *args, **kwargs):
        super(CorrelateDp4aBlock, self).__init__(iring, *args, **kwargs)
        self.nframe_to_avg = nframe_to_avg
        self.frame_count    = 0

    def define_valid_input_spaces(self):
        """Return set of valid spaces (or 'any') for each input"""
        return ('cuda',)
    
    def on_sequence(self, iseq):
        self.frame_count = 0
        ihdr = iseq.header
        itensor = ihdr['_tensor']
        try:
            assert(itensor['labels'] == ['time', 'freq', 'station', 'fine_time'])
            assert(itensor['dtype'] == 'ci8')
            assert(itensor['shape'][2] <= 32)   # This kernel only works if N<=32 
            if self.nframe_to_avg > 1:
                assert(ihdr['gulp_nframe'] == 1)
        except AssertionError:
            print('ERR: gulp_nframe %s' % str(ihdr['gulp_nframe']))
            print('ERR: Frame shape %s' % str(itensor['shape']))
            print('ERR: Frame labels %s' % str(itensor['labels']))
            print('ERR: Frame dtype %s' % itensor['dtype'])
            raise RuntimeError('Correlator block misconfiguration. Check tensor labels, dtype, shape, gulp size).')


        ohdr = deepcopy(ihdr)
        otensor = ohdr['_tensor']
        otensor['dtype'] = 'cf32'
        
        for key in ['shape', 'labels', 'scales', 'units']:
            time_val, freq_val, stand_val, ftime_val = itensor[key]
            otensor[key] = [time_val, freq_val, stand_val, stand_val]
        
        # Take into account averaging across frames / gulp size
        otensor['scales'][0][1] *= self.nframe_to_avg  
        otensor['labels'][2] += '_i'
        otensor['labels'][3] += '_j'
    
        return ohdr

    def on_data(self, ispan, ospan):
        idata = ispan.data
        odata = ospan.data
        reset = 0
        if self.frame_count == 0:
            reset = 1   # Reset output array to zero before computing xcorr
        
        # Run the cross-correlation
        _bf.XcorrLite(idata.as_BFarray(), odata.as_BFarray(), np.int32(reset))

        # Update gulp / frame counters and reset as required
        self.frame_count += 1   
        if self.frame_count == self.nframe_to_avg:
            ncommit = ispan.data.shape[0]
            self.frame_count = 0
        else:
            ncommit = 0
        return ncommit

def correlate_dp4a(iring, nframe_to_avg=1, *args, **kwargs):
    """ Cross-correlate [time freq station fine_time] array using GPU.

    Simple GPU kernel for up to N=32 antennas, uses DP4A instruction for
    cross multiply and accumulate.

    ** Tensor Semantics **
    Input:  [time, freq, station, fine_time]
    Output: [time, freq, station_i, station_j]

    Notes: Averages across fine_time. Can also average across 
    gulp_nframe axis ('time').

    Limitations:
      * Requires 8-bit complex data input
      * fine_time axis must have len <= 4096 to avoid int32 overflow in DP4a
      * station axis needs to be < 32. (the simple GPU kernel requires N <= 32)
      * Can currently only average across frames if gulp_nframe = 1

    Args:
      nframe_to_avg (int): Number of frames to average across. 1 = no averaging.
      iring (Ring or Block): Input data source.
      *args: Arguments to ``bifrost.pipeline.TransformBlock``.
      **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.
    Returns:
        CorrelateDp4aBlock: A new correlator block instance.
    """
    return CorrelateDp4aBlock(iring, nframe_to_avg, *args, **kwargs)


