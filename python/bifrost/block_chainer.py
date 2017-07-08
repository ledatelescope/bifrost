
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

import bifrost

class _BlockChainerProxy(object):
    def __init__(self, parent, module):
        self.parent = parent
        self.module = module
    def __getattr__(self, attr):
        func = getattr(self.module, attr)
        return self.parent._get(func)

class BlockChainer(object):
    """Convenient tool for constructing linear chains of blocks and views

    Examples::

        bc = bf.BlockChainer()
        bc.blocks.read_sigproc("foo.fil", gulp_nframe=1)
        bc.blocks.copy('cuda')
        bc.views.split_axis('freq', 2, 'fine_freq')
        bc.views.merge_axes('freq', 'fine_freq')
        bc.blocks.copy('cuda_host')
        bc.custom(my_block)(arg1, arg2, ...)
        bc.blocks.write_sigproc()
        print bc.last_block # The last added block (this can also be set)
    """
    @property
    def blocks(self):
        return _BlockChainerProxy(self, bifrost.blocks)
    @property
    def views(self):
        return _BlockChainerProxy(self, bifrost.views)
    def custom(self, func):
        return self._get(func)
    def _get(self, func):
        def wrapper(*args, **kwargs):
            if hasattr(self, 'last_block'):
                self.last_block = func(self.last_block, *args, **kwargs)
            else:
                self.last_block = func(*args, **kwargs)
            return self.last_block
        return wrapper
