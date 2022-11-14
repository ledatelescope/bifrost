
# Copyright (c) 2022, The Bifrost Authors. All rights reserved.
# Copyright (c) 2022, The University of New Mexico. All rights reserved.
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

# Python2 compatibility
from __future__ import absolute_import

from ctypes import c_ulong, pointer as c_pointer

from bifrost.libbifrost import _bf, _check, BifrostObject, _string2space
from bifrost.ndarray import asarray

from bifrost import telemetry
telemetry.track_module()

class Rayleigh(BifrostObject):
    def __init__(self):
        BifrostObject.__init__(self, _bf.bfRayleighCreate, _bf.bfRayleighDestroy)
    def init(self, nantpols, alpha=0.75, clip_sigmas=6, max_flag_frac=0.25, space='cuda'):
        space = _string2space(space)
        psize = None
        self._clip_sigmas = clip_sigmas
        self._max_flag_frac = max_flag_frac
        _check( _bf.bfRayleighInit(self.obj, nantpols, alpha, clip_sigmas*clip_sigmas, max_flag_frac, space, 0, psize) )
    def reset_state(self):
        _check( _bf.bfRayleighResetState(self.obj) )
    def execute(self, idata, odata):
        # TODO: Work out how to integrate CUDA stream
        flags = c_ulong(0)
        _check( _bf.bfRayleighExecute(self.obj,
                                      asarray(idata).as_BFarray(),
                                      asarray(odata).as_BFarray(),
                                      c_pointer(flags)) )
        return flags.value, odata
