
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
import bifrost.unpack
from bifrost.pipeline import TransformBlock
from bifrost.DataType import DataType

from copy import deepcopy

class UnpackBlock(TransformBlock):
    def __init__(self, iring, dtype, align_msb=False,
                 *args, **kwargs):
        super(UnpackBlock, self).__init__(iring, *args, **kwargs)
        self.dtype     = dtype
        self.align_msb = align_msb
    def define_valid_input_spaces(self):
        """Return set of valid spaces (or 'any') for each input"""
        return ('system',)
    def on_sequence(self, iseq):
        ihdr = iseq.header
        ohdr = deepcopy(ihdr)
        itype = DataType(ihdr['_tensor']['dtype'])
        self.itype = itype
        # Allow user to pass nbit instead of explicit dtype
        if isinstance(self.dtype, int):
            nbit = self.dtype
            otype = itype.as_nbit(nbit)
        else:
            otype = self.dtype
        ohdr['_tensor']['dtype'] = otype
        return ohdr
    def on_data(self, ispan, ospan):
        idata = ispan.data
        odata = ospan.data
        bf.unpack.unpack(idata, odata, self.align_msb)

def unpack(iring, dtype, *args, **kwargs):
    """Unpack data to a larger data type.

    Args:
        iring (Ring or Block): Input data source.
        dtype: Output data type.
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.

    **Tensor semantics**::

        Input:  [...], dtype = one of: i/u2, i/u4, ci2, ci4, space = SYSTEM
        Output: [...], dtype = i8 or ci8 (matching input), space = SYSTEM

    Returns:
        UnpackBlock: A new block instance.
    """
    return UnpackBlock(iring, dtype, *args, **kwargs)
