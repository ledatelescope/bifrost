
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

class Fft(BifrostObject):
    def __init__(self):
        BifrostObject.__init__(self, _bf.bfFftCreate, _bf.bfFftDestroy)
    def init(self, iarray, oarray, axes=None, apply_fftshift=False):
        if isinstance(axes, int):
            axes = [axes]
        ndim = len(axes)
        if axes is not None:
            axes_type = ctypes.c_int * ndim
            axes = axes_type(*axes)
        self.workspace_size = _get(_bf.bfFftInit,
                                   self.obj,
                                   asarray(iarray).as_BFarray(),
                                   asarray(oarray).as_BFarray(),
                                   ndim,
                                   axes,
                                   apply_fftshift)
    def execute(self, iarray, oarray, inverse=False):
        return self.execute_workspace(iarray, oarray,
                                      workspace_ptr=None, workspace_size=0,
                                      inverse=inverse)
    def execute_workspace(self, iarray, oarray, workspace_ptr, workspace_size,
                          inverse=False):
        _check(_bf.bfFftExecute(
            self.obj,
            asarray(iarray).as_BFarray(),
            asarray(oarray).as_BFarray(),
            inverse,
            workspace_ptr, workspace_size))
        return oarray
