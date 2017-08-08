
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

from libbifrost import _bf, _check, _get, BifrostObject, _string2space
from ndarray import asarray

import ctypes
import numpy as np

class Fdmt(BifrostObject):
    def __init__(self):
        BifrostObject.__init__(self, _bf.bfFdmtCreate, _bf.bfFdmtDestroy)
    def init(self, nchan, max_delay, f0, df, exponent=-2.0, space='cuda'):
        space = _string2space(space)
        psize = None
        _check(_bf.bfFdmtInit(self.obj, nchan, max_delay, f0, df,
                              exponent, space, 0, psize))
    def execute(self, idata, odata, negative_delays=False):
        # TODO: Work out how to integrate CUDA stream
        psize = None
        _check( _bf.bfFdmtExecute(
            self.obj,
            asarray(idata).as_BFarray(),
            asarray(odata).as_BFarray(),
            negative_delays,
            None,
            psize) )
        return odata
    def get_workspace_size(self, idata, odata):
        return _get(_bf.bfFdmtExecute,
                    self.obj,
                    asarray(idata).as_BFarray(),
                    asarray(odata).as_BFarray(),
                    False,
                    None)
    def execute_workspace(self, idata, odata, workspace_ptr, workspace_size,
                          negative_delays=False):
        size = _bf.BFsize(workspace_size)
        _check(_bf.bfFdmtExecute(
            self.obj,
            asarray(idata).as_BFarray(),
            asarray(odata).as_BFarray(),
            negative_delays,
            workspace_ptr,
            size))
        return odata
