
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

import ctypes
from functools import reduce


class _CaptureBase(BifrostObject):
    @staticmethod
    def _flatten_value(value):
        try:
            value = reduce(lambda x,y: x*y, value, 1 if value else 0)
        except TypeError:
            pass
        return value
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.end()
    def recv(self):
        status = _bf.BFpacketcapture_status()
        _check(_bf.bfPacketCaptureRecv(self.obj, status))
        return status.value
    def flush(self):
        _check(_bf.bfPacketCaptureFlush(self.obj))
    def end(self):
        _check(_bf.bfPacketCaptureEnd(self.obj))
    def set_callback(self, fnc):
        self._seq_callback = _bf.BFpacketcapture_drx_sequence_callback(fnc)
        _check(_bf.bfPacketCaptureSetCallback(
               self.obj, self._seq_callback))

class UDPCapture(_CaptureBase):
    def __init__(self, fmt, sock, ring, nsrc, src0, max_payload_size,
                 buffer_ntime, slot_ntime, core=None):
        try:
            fmt = fmt.encode()
        except AttributeError:
            # Python2 catch
            pass
        nsrc = self._flatten_value(nsrc)
        if core is None:
            core = -1
        BifrostObject.__init__(
            self, _bf.bfUdpCaptureCreate, _bf.bfPacketCaptureDestroy,
            fmt, sock.fileno(), ring.obj, nsrc, src0,
            max_payload_size, buffer_ntime, slot_ntime,
            core)

class UDPSniffer(_CaptureBase):
    def __init__(self, fmt, sock, ring, nsrc, src0, max_payload_size,
                 buffer_ntime, slot_ntime, core=None):
        try:
            fmt = fmt.encode()
        except AttributeError:
            # Python2 catch
            pass
        nsrc = self._flatten_value(nsrc)
        if core is None:
            core = -1
        BifrostObject.__init__(
            self, _bf.bfUdpSnifferCreate, _bf.bfPacketCaptureDestroy,
            fmt, sock.fileno(), ring.obj, nsrc, src0,
            max_payload_size, buffer_ntime, slot_ntime,
            core)

class UDPVerbsCapture(_CaptureBase):
    def __init__(self, fmt, sock, ring, nsrc, src0, max_payload_size,
                 buffer_ntime, slot_ntime, core=None):
        try:
            fmt = fmt.encode()
        except AttributeError:
            # Python2 catch
            pass
        if core is None:
            core = -1
        BifrostObject.__init__(
            self, _bf.bfUdpVerbsCaptureCreate, _bf.bfPacketCaptureDestroy,
            fmt, sock.fileno(), ring.obj, nsrc, src0,
            max_payload_size, buffer_ntime, slot_ntime,
            core)

class DiskReader(_CaptureBase):
    def __init__(self, fmt, fh, ring, nsrc, src0,
                 buffer_nframe, slot_nframe, core=None):
        try:
            fmt = fmt.encode()
        except AttributeError:
            # Python2 catch
            pass
        nsrc = self._flatten_value(nsrc)
        if core is None:
            core = -1
        BifrostObject.__init__(
            self, _bf.bfDiskReaderCreate, _bf.bfPacketCaptureDestroy,
            fmt, fh.fileno(), ring.obj, nsrc, src0,
            buffer_nframe, slot_nframe,
            core)
        # Make sure we start in the same place in the file
        self.seek(fh.tell(), _bf.BF_WHENCE_SET)
    def seek(self, offset, whence=_bf.BF_WHENCE_CUR):
        position = ctypes.c_ulong(0)
        _check(_bf.bfPacketCaptureSeek(self.obj, offset, whence, position))
        return position.value
    def tell(self):
        position = ctypes.c_ulong(0)
        _check(_bf.bfPacketCaptureTell(self.obj, position))
        return position.value
