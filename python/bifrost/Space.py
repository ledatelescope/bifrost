
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

from bifrost.libbifrost import _bf

SPACEMAP_TO_STR = {_bf.BF_SPACE_AUTO:         'auto',
                   _bf.BF_SPACE_SYSTEM:       'system',
                   _bf.BF_SPACE_CUDA:         'cuda',
                   _bf.BF_SPACE_CUDA_HOST:    'cuda_host',
                   _bf.BF_SPACE_CUDA_MANAGED: 'cuda_managed'}

SPACEMAP_FROM_STR = {'auto':         _bf.BF_SPACE_AUTO,
                     'system':       _bf.BF_SPACE_SYSTEM,
                     'cuda':         _bf.BF_SPACE_CUDA,
                     'cuda_host':    _bf.BF_SPACE_CUDA_HOST,
                     'cuda_managed': _bf.BF_SPACE_CUDA_MANAGED}

class Space(object):
    def __init__(self, s):
        if isinstance(s, basestring):
            if s not in set(['auto', 'system',
                             'cuda', 'cuda_host', 'cuda_managed']):
                raise ValueError('Invalid space: %s' % s)
            self._space = s
        elif isinstance(s, _bf.BFspace) or isinstance(s, int):
            if s not in SPACEMAP_TO_STR:
                raise KeyError("Invalid space: " + s +
                               ". Valid spaces: " + str(SPACEMAP_TO_STR.keys()))
            self._space = SPACEMAP_TO_STR[s]
        else:
            raise ValueError('%s is not a space' % s)
    def as_BFspace(self):
            return SPACEMAP_FROM_STR[self._space]
    def __str__(self):
        return self._space
