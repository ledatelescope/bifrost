
# Copyright (c) 2016-2023, The Bifrost Authors. All rights reserved.
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

from bifrost.quantize import quantize as bf_quantize
from bifrost.pipeline import TransformBlock
from bifrost.DataType import DataType
from bifrost.ring2 import Ring, ReadSequence, ReadSpan, WriteSpan

from copy import deepcopy
import numpy as np

from typing import Any, Dict, Union, Tuple

from bifrost import telemetry
telemetry.track_module()

class QuantizeBlock(TransformBlock):
    def __init__(self, iring: Ring, dtype: Union[str,np.dtype], scale: float=1.,
                 *args, **kwargs):
        super(QuantizeBlock, self).__init__(iring, *args, **kwargs)
        self.dtype = dtype
        self.scale = scale
    def define_valid_input_spaces(self) -> Tuple[str]:
        """Return set of valid spaces (or 'any') for each input"""
        return ('any',)
    def on_sequence(self, iseq: ReadSequence) -> Dict[str,Any]:
        ihdr = iseq.header
        ohdr = deepcopy(ihdr)
        itype = DataType(ihdr['_tensor']['dtype'])
        self.itype = itype
        # Allow user to pass nbit instead of explicit dtype
        if isinstance(self.dtype, int):
            nbit = self.dtype
            otype = itype.as_integer(nbit)
        else:
            otype = self.dtype
        ohdr['_tensor']['dtype'] = otype
        return ohdr
    def on_data(self, ispan: ReadSpan, ospan: WriteSpan) -> None:
        idata = ispan.data
        odata = ospan.data
        bf_quantize(idata, odata, self.scale)

def quantize(iring: Ring, dtype: Union[str,np.dtype], scale: float=1., *args, **kwargs) -> QuantizeBlock:
    """Apply a requantization of bit depth for the data.

    Args:
        iring (Ring or Block): Input data source.
        dtype: Output data type or number of bits.
        scale (float): Scale factor to apply before quantizing.
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.

    **Tensor semantics**::

        Input:  [...], dtype = [c]f32, space = SYSTEM or CUDA
        Output: [...], dtype = any (complex) integer type, space = SYSTEM or CUDA

    Returns:
        QuantizeBlock: A new block instance.
    """
    return QuantizeBlock(iring, dtype, scale, *args, **kwargs)
