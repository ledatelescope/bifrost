
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

# **TODO: Write tests for this class

from libbifrost import _bf, _check, _get, BifrostObject

class UDPSocket(BifrostObject):
    def __init__(self):
        BifrostObject.__init__(self, _bf.bfUdpSocketCreate, _bf.bfUdpSocketDestroy)
    def bind(self, local_addr):
        _check( _bf.bfUdpSocketBind(self.obj, local_addr.obj) )
    def connect(self, remote_addr):
        _check( _bf.bfUdpSocketConnect(self.obj, remote_addr.obj) )
    def shutdown(self):
        _check( _bf.bfUdpSocketShutdown(self.obj) )
    def close(self):
        _check( _bf.bfUdpSocketClose(self.obj) )
    @property
    def mtu(self):
        return _get(_bf.bfUdpSocketGetMTU, self.obj)
    def fileno(self):
        return _get(_bf.bfUdpSocketGetFD, self.obj)
    @property
    def timeout(self):
        return _get(_bf.bfUdpSocketGetTimeout, self.obj)
    @timeout.setter
    def timeout(self, secs):
        _check( _bf.bfUdpSocketSetTimeout(self.obj, secs) )
