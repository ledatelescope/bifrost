
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

from libbifrost import _bf, _check, _get, BifrostObject

import ctypes
from socket import AF_UNSPEC

class Address(BifrostObject):
    def __init__(self, address, port, family=None):
        assert(isinstance(port, (int, long)))
        if family is None:
            family = AF_UNSPEC
        BifrostObject.__init__(
            self, _bf.bfAddressCreate, _bf.bfAddressDestroy,
            address, port, family)
    @property
    def family(self):
        return _get(_bf.bfAddressGetFamily, self.obj)
    @property
    def port(self):
        return _get(_bf.bfAddressGetPort, self.obj)
    @property
    def mtu(self):
        return _get(_bf.bfAddressGetMTU, self.obj)
    @property
    def address(self):
        buflen = 128
        buf = ctypes.create_string_buffer(buflen)
        _check(_bf.bfAddressGetString(self.obj, buflen, buf))
        return buf.value
    def __str__(self):
        return "%s:%i" % (self.address, self.port)
