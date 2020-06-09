
# Copyright (c) 2019-2020, The Bifrost Authors. All rights reserved.
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

from bifrost.libbifrost import _bf, _check, _get, BifrostObject
from bifrost.ndarray import asarray

class HeaderInfo(BifrostObject):
    def __init__(self):
        BifrostObject.__init__(
            self, _bf.bfHeaderInfoCreate, _bf.bfHeaderInfoDestroy)
    def set_nsrc(self,  nsrc):
        _check(_bf.bfHeaderInfoSetNSrc(self.obj, nsrc))
    def set_nchan(self, nchan):
        _check(_bf.bfHeaderInfoSetNChan(self.obj, nchan))
    def set_chan0(self, chan0):
        _check(_bf.bfHeaderInfoSetChan0(self.obj, chan0))
    def set_tuning(self, tuning):
        _check(_bf.bfHeaderInfoSetTuning(self.obj, tuning))
    def set_gain(self, gain):
        _check(_bf.bfHeaderInfoSetGain(self.obj, gain))
    def set_decimation(self, decimation):
        _check(_bf.bfHeaderInfoSetDecimation(self.obj, decimation))

class _WriterBase(BifrostObject):
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        pass
    def reset_counter(self):
        _check(_bf.bfPacketWriterResetCounter(self.obj))
    def send(self, headerinfo, seq, seq_increment, src, src_increment, idata):
        _check(_bf.bfPacketWriterSend(self.obj,
                                      headerinfo.obj,
                                      seq,
                                      seq_increment,
                                      src,
                                      src_increment,
                                      asarray(idata).as_BFarray()))

class UDPTransmit(_WriterBase):
    def __init__(self, fmt, sock, core=None):
        try:
            fmt = fmt.encode()
        except AttributeError:
            # Python2 catch
            pass
        if core is None:
            core = -1
        BifrostObject.__init__(
            self, _bf.bfUdpTransmitCreate, _bf.bfPacketWriterDestroy,
            fmt, sock.fileno(), core)
   

class DiskWriter(_WriterBase):
    def __init__(self, fmt, fh, core=None):
        try:
            fmt = fmt.encode()
        except AttributeError:
            # Python2 catch
            pass
        if core is None:
            core = -1
        BifrostObject.__init__(
            self, _bf.bfDiskWriterCreate, _bf.bfPacketWriterDestroy,
            fmt, fh.fileno(), core)
