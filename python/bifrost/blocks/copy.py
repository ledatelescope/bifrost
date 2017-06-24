
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
from bifrost.ndarray import copy_array

from copy import deepcopy

class CopyBlock(TransformBlock):
    def __init__(self, iring, space=None, *args, **kwargs):
        super(CopyBlock, self).__init__(iring, *args, **kwargs)
        if space is None:
            space = self.iring.space
        self.orings = [self.create_ring(space=space)]
    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        return ohdr
    def on_data(self, ispan, ospan):
        copy_array(ospan.data, ispan.data)

def copy(iring, space=None, *args, **kwargs):
    """Copy data, possibly to another space.

    Use this block to copy data between different
        spaces, such as from system memory to GPU memory.
        The output header for this block is identical
        to the input header.

    Args:
        iring (Ring or Block): Input data source.
        space (str): Output data space (e.g., 'cuda' or 'system').
            Default space is same as input.
        *args:  Arguments to ``bifrost.pipeline.TransformBlock``.
        *kwargs: Keyword arguments to ``bifrost.pipeline.TransformBlock``.

    Returns:
        CopyBlock: A new block instance.

    **Tensor semantics**::

            Input:  [...], dtype = any, space = any
            Output: [...], dtype = same as input, space = any
    """
    return CopyBlock(iring, space, *args, **kwargs)
