
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

from copy import deepcopy

class FftShiftBlock(TransformBlock):
    def __init__(self, iring, axes, inverse=False,
                 *args, **kwargs):
        super(FftShiftBlock, self).__init__(iring, *args, **kwargs)
        if not isinstance(axes, list) or isinstance(axes, tuple):
            axes = [axes]
        self.specified_axes = axes
        self.inverse = inverse
    def define_valid_input_spaces(self):
        return ('cuda',)
    def on_sequence(self, iseq):
        ihdr = iseq.header
        itensor = ihdr['_tensor']
        self.axes = [itensor['labels'].index(axis)
                     if isinstance(axis, basestring)
                     else axis
                     for axis in self.specified_axes]
        frame_axis = itensor['shape'].index(-1)
        if frame_axis in self.axes:
            raise KeyError("Cannot fftshift frame axis")
        ohdr = deepcopy(ihdr)
        otensor = ohdr['_tensor']
        oshape = otensor['shape']
        if 'scales' in itensor:
            for ax in self.axes:
                sgn = +1 if self.inverse else -1
                scale_step = otensor['scales'][ax][1]
                # TODO: Double-check that this does what we want when scale_delta is negative
                scale_shift = sgn * (oshape[ax] // 2) * scale_step
                otensor['scales'][ax][0] += scale_shift
        return ohdr
    def on_data(self, ispan, ospan):
        idata = ispan.data
        odata = ospan.data
        shape = idata.shape
        ind_names = ['i%i' % i for i in xrange(idata.ndim)]
        inds = list(ind_names)
        for ax in self.axes:
            if self.inverse:
                inds[ax] += '-(a.shape(%i)-a.shape(%i)/2)' % (ax, ax)
            else:
                inds[ax] += '-a.shape(%i)/2' % ax
        inds = ','.join(inds)
        bf.map("b = a(%s)" % inds, shape=shape, axis_names=ind_names,
               data={'a': idata, 'b': odata})

def fftshift(iring, axes, inverse=False, *args, **kwargs):
    """Apply an FFT shift to data along specified axes.

    This operation shifts the data of an array so the central element
    is moved to the start of the array.

    Args:
        iring (Ring or Block): Input data source.
        axes: (List of) strings or integers specifying axes to shift.
        inverse (bool): If True, axes are shifted in the reverse direction.
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.

    **Tensor semantics**::

        Input:  [...], dtype = any, space = CUDA
        Output: [...], dtype = any, space = CUDA

    Returns:
        FftShiftBlock: A new block instance.
    """
    return FftShiftBlock(iring, axes, inverse, *args, **kwargs)
