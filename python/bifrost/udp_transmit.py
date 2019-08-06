# -*- coding: utf-8 -*-
# Copyright (c) 2017, The Bifrost Authors. All rights reserved.
# Copyright (c) 2017, The University of New Mexico. All rights reserved.
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

import ctypes

def _packet2pointer(packet):
    buf = ctypes.c_char_p(packet)
    siz = ctypes.c_uint( len(packet) )
    return buf, siz

def _packets2pointer(packets):
    count = ctypes.c_uint( len(packets) )
    buf = ctypes.c_char_p("".join(packets))
    siz = ctypes.c_uint( len(packets[0]) )
    return buf, siz, count

class UDPTransmit(BifrostObject):
    def __init__(self, sock, core=-1):
        BifrostObject.__init__(
            self, _bf.bfUdpTransmitCreate, _bf.bfUdpTransmitDestroy,
            sock.fileno(), core)
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        pass
    def send(self, packet):
        ptr, siz = _packet2pointer(packet)
        _check(_bf.bfUdpTransmitSend(self.obj, ptr, siz))
    def sendmany(self, packets):
        assert(type(packets) is list)
        ptr, siz, count = _packets2pointer(packets)
        _check(_bf.bfUdpTransmitSendMany(self.obj, ptr, siz, count))
