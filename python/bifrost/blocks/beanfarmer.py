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
import hickle as hkl

from copy import deepcopy


class BeanfarmerDp4aBlock(bf.pipeline.TransformBlock):
    def __init__(self, iring, n_avg=1, n_beam=32, n_chan=512, n_pol=2, n_ant=12, weights_file='',
                 *args, **kwargs):
        super(BeanfarmerDp4aBlock, self).__init__(iring, *args, **kwargs)
        self.n_avg  = n_avg
        self.n_beam = n_beam
        self.n_pol  = n_pol
        self.n_chan = n_chan
        self.n_ant  = n_ant
        self.frame_count    = 0
        self.weights_file = weights_file

    def define_valid_input_spaces(self):
        """Return set of valid spaces (or 'any') for each input"""
        return ('cuda',)
    
    def on_sequence(self, iseq):
        self.frame_count = 0
        ihdr = iseq.header
        itensor = ihdr['_tensor']

        to_raise = False
        
        if self.weights_file in ('', None):
            to_raise = True
            print('ERR: need to specify weights hickle file')
        else:
            w = hkl.load(self.weights_file)

        try:
            assert w.shape == (self.n_chan, self.n_beam, self.n_pol, self.n_ant, 2)
            assert w.dtype.str == '|i1'
        except AssertionError:
            print('ERR: beam weight shape/dtype is incorrect')
            print('ERR: beam weights shape is: %s' % str(w.shape))
            print('ERR: shape should be %s' % str((self.n_chan, self.n_beam, self.n_pol, self.n_ant, 2)))
            print('ERR: dtype should be int8, dtype: %s' % w.dtype.str)
            to_raise = True
        #w = np.ones((self.n_chan, self.n_beam, self.n_pol, self.n_ant), dtype='int8')
        self.weights = bf.ndarray(w, dtype='ci8', space='cuda')

        try:
            assert(itensor['labels'] == ['time', 'freq', 'fine_time', 'pol', 'station'])
            assert(itensor['dtype'] == 'ci8')
            assert(ihdr['gulp_nframe'] == 1)
        except AssertionError:
            print('ERR: gulp_nframe %s (must be 1!)' % str(ihdr['gulp_nframe']))
            print('ERR: Frame shape %s' % str(itensor['shape']))
            print('ERR: Frame labels %s' % str(itensor['labels']))
            print('ERR: Frame dtype %s' % itensor['dtype'])
            to_raise = True
        
        if to_raise:
            raise RuntimeError('Correlator block misconfiguration. Check tensor labels, dtype, shape, gulp size).')

        ohdr = deepcopy(ihdr)
        otensor = ohdr['_tensor']
        otensor['dtype'] = 'cf32'

        # output is (time, channel, beam, fine_time)
        ft0, fts = itensor['scales'][2]
        otensor['shape']  = [itensor['shape'][0], itensor['shape'][1], self.n_beam, itensor['shape'][2] // self.n_avg]
        otensor['labels'] = ['time', 'freq', 'beam', 'fine_time']
        otensor['scales'] = [itensor['scales'][0], itensor['scales'][1], [0, 0], [ft0, fts / self.n_avg]]
        otensor['units']  = [itensor['units'][0], itensor['units'][1], None, itensor['units'][2]]
        otensor['dtype'] = 'f32'

        return ohdr

    def on_data(self, ispan, ospan):
        idata = ispan.data
        odata = ospan.data

        # Run the beamformer
        #print(idata.shape, self.weights.shape, odata.shape, self.n_avg)
        res = _bf.BeanFarmer(idata.as_BFarray(), self.weights.as_BFarray(), odata.as_BFarray(), np.int32(self.n_avg))

        ncommit = ispan.data.shape[0]
        return ncommit


def beanfarmer(iring, n_avg=1, n_beam=32, n_chan=512, n_pol=2, n_ant=12, weights_file='', *args, **kwargs):
    """ Beamform, detect + integrate (filterbank) array using GPU.

    ** Tensor Semantics **
    Input:  [time, freq, fine_time, pol, station]
    Output: [time, freq, beam, fine_time]

    Notes: Averages across fine_time.

    Limitations:
      * Requires 8-bit complex data input
      * Currently only works if gulp_nframe = 1

    Args:
      nframe_to_avg (int): Number of frames to average across. 1 = no averaging.
      iring (Ring or Block): Input data source.
      n_avg (int): Number of frames to average together
      n_beam (int): Number of beams to form
      n_chan (int): Number of channels
      n_pol  (int): Number of polarizations for antennas (1 or 2)
      n_ant  (int): Number of antennas/stands (n_ant=12 and n_pol=2 means 24 inputs)
      weights_file (str): Path to hickle file in which beam weights are stored. Beam weights
                          must have the same shape as (chan, pol, ant, beam) etc here.
      *args: Arguments to ``bifrost.pipeline.TransformBlock``.
      **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.
    Returns:
        CorrelateDp4aBlock: A new correlator block instance.
    """

    return BeanfarmerDp4aBlock(iring, n_avg, n_beam, n_chan, n_pol, n_ant, weights_file, *args, **kwargs)


