
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

# TODO: Consider merging with detect_block
#         Seems easy, but may end up somewhat complicated

from __future__ import absolute_import

from bifrost.pipeline import TransformBlock
import bifrost as bf

from copy import deepcopy

class ReduceBlock(TransformBlock):
    def __init__(self, iring, axis, factor=None, op='sum', *args, **kwargs):
        super(ReduceBlock, self).__init__(iring, *args, **kwargs)
        self.specified_axis   = axis
        self.specified_factor = factor
        self.op = op
    def define_valid_input_spaces(self):
        return ('cuda',)
    def on_sequence(self, iseq):
        ihdr = iseq.header
        itensor = ihdr['_tensor']
        ohdr = deepcopy(ihdr)
        otensor = ohdr['_tensor']
        otensor['dtype'] = 'f32'
        if 'labels' in itensor and isinstance(self.specified_axis, basestring):
            # Look up axis by label
            self.axis = itensor['labels'].index(self.specified_axis)
        else:
            self.axis = self.specified_axis
        self.frame_axis = itensor['shape'].index(-1)
        self.factor = self.specified_factor
        if self.axis == self.frame_axis:
            if self.specified_factor is None:
                raise ValueError("Reduce factor must be specified for frame axis")
        else:
            if self.specified_factor is None:
                # Default to reducing the whole axis
                self.factor = otensor['shape'][self.axis]
            elif otensor['shape'][self.axis] % self.factor != 0:
                raise ValueError("Reduce factor does not divide axis length")
            otensor['shape'][self.axis] //= self.factor
        otensor['scales'][self.axis][1] *= self.factor
        return ohdr
    def define_output_nframes(self, input_nframe):
        output_nframe = input_nframe
        if self.axis == self.frame_axis:
            if input_nframe % self.factor != 0:
                raise ValueError("Reduce factor does not divide input_nframe")
            output_nframe = input_nframe // self.factor
        return output_nframe
    def on_data(self, ispan, ospan):
        idata, odata = ispan.data, ospan.data
        bf.reduce(idata, odata, self.op)
        
        # TODO: Support system space using Numpy
        #ishape = list(idata.shape)
        #ishape[self.axis] //= self.factor
        #ishape.insert(self.axis+1, self.factor)

def reduce(iring, axis, factor=None, op='sum', *args, **kwargs):
    """Reduce data along an axis by factor using op.

    Args:
        iring (Ring or Block): Input data source.
        axis (int or str): The axis to reduce. Can be an integer index
                           or a string label.
        factor (int): The factor by which the axis should be reduced.
                      If None, the whole axis is reduced. Must divide
                      the size of the axis (or the gulp_size in the case
                      where the axis is the frame axis).
        op (str): The operation with which the data should be reduced.
                  One of: sum, mean, min, max, stderr [stderr=sum/sqrt(n)].
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.

    **Tensor semantics**::

        Input:  [..., N, ...], dtype = any real, space = CUDA
        Output: [..., N / factor, ...], dtype = f32, space = CUDA

    Returns:
        ReduceBlock: A new block instance.
    """
    return ReduceBlock(iring, axis, factor, op, *args, **kwargs)
