
# Copyright (c) 2019-2024, The Bifrost Authors. All rights reserved.
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

from bifrost.libbifrost import _bf, _th, _check, _get, BifrostObject
from bifrost.udp_socket import UDPSocket
from bifrost.ring import Ring
from bifrost.ring2 import Ring as Ring2

import ctypes
from functools import reduce
from io import IOBase

from typing import Optional, Union

from bifrost import telemetry
telemetry.track_module()

class PacketCaptureCallback(BifrostObject):
    _ref_cache = {}
    def __init__(self):
        BifrostObject.__init__(
            self, _bf.bfPacketCaptureCallbackCreate, _bf.bfPacketCaptureCallbackDestroy)

    def set_simple(self, fnc):
        self._ref_cache['simple'] = _bf.BFpacketcapture_simple_sequence_callback(fnc)
        _check(_bf.bfPacketCaptureCallbackSetSIMPLE(
               self.obj, self._ref_cache['simple']))

    def set_chips(self, fnc: _bf.BFpacketcapture_chips_sequence_callback):
        self._ref_cache['chips'] = _bf.BFpacketcapture_chips_sequence_callback(fnc)
        _check(_bf.bfPacketCaptureCallbackSetCHIPS(
               self.obj, self._ref_cache['chips']))
    def set_snap2(self, fnc: _bf.BFpacketcapture_snap2_sequence_callback):
        self._ref_cache['snap2'] = _bf.BFpacketcapture_snap2_sequence_callback(fnc)
        _check(_bf.bfPacketCaptureCallbackSetSNAP2(
               self.obj, self._ref_cache['snap2']))
    def set_ibeam(self, fnc: _bf.BFpacketcapture_ibeam_sequence_callback):
        self._ref_cache['ibeam'] = _bf.BFpacketcapture_ibeam_sequence_callback(fnc)
        _check(_bf.bfPacketCaptureCallbackSetIBeam(
               self.obj, self._ref_cache['ibeam']))
    def set_pbeam(self, fnc: _bf.BFpacketcapture_pbeam_sequence_callback):
        self._ref_cache['pbeam'] = _bf.BFpacketcapture_pbeam_sequence_callback(fnc)
        _check(_bf.bfPacketCaptureCallbackSetPBeam(
               self.obj, self._ref_cache['pbeam']))
    def set_cor(self, fnc: _bf.BFpacketcapture_cor_sequence_callback):
        self._ref_cache['cor'] = _bf.BFpacketcapture_cor_sequence_callback(fnc)
        _check(_bf.bfPacketCaptureCallbackSetCOR(
               self.obj, self._ref_cache['cor']))
    def set_vdif(self, fnc: _bf.BFpacketcapture_vdif_sequence_callback):
        self._ref_cache['vdif'] = _bf.BFpacketcapture_vdif_sequence_callback(fnc)
        _check(_bf.bfPacketCaptureCallbackSetVDIF(
               self.obj, self._ref_cache['vdif']))
    def set_tbn(self, fnc: _bf.BFpacketcapture_tbn_sequence_callback):
        self._ref_cache['tbn'] = _bf.BFpacketcapture_tbn_sequence_callback(fnc)
        _check(_bf.bfPacketCaptureCallbackSetTBN(
               self.obj, self._ref_cache['tbn']))
    def set_drx(self, fnc: _bf.BFpacketcapture_drx_sequence_callback):
        self._ref_cache['drx'] = _bf.BFpacketcapture_drx_sequence_callback(fnc)
        _check(_bf.bfPacketCaptureCallbackSetDRX(
            self.obj, self._ref_cache['drx']))
    def set_drx8(self, fnc: _bf.BFpacketcapture_drx8_sequence_callback):
        self._ref_cache['drx8'] = _bf.BFpacketcapture_drx8_sequence_callback(fnc)
        _check(_bf.bfPacketCaptureCallbackSetDRX8(
            self.obj, self._ref_cache['drx8']))

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
    def recv(self) -> _th.BFcapture_enum:
        status = _bf.BFpacketcapture_status()
        _check(_bf.bfPacketCaptureRecv(self.obj, status))
        return status.value
    def flush(self):
        _check(_bf.bfPacketCaptureFlush(self.obj))
    def end(self):
        _check(_bf.bfPacketCaptureEnd(self.obj))

class UDPCapture(_CaptureBase):
    def __init__(self, fmt: str, sock: UDPSocket, ring: Union[Ring,Ring2],
                 nsrc: int, src0: int, max_payload_size: int, buffer_ntime: int,
                 slot_ntime: int, sequence_callback: PacketCaptureCallback,
                 core: Optional[int]=None):
        nsrc = self._flatten_value(nsrc)
        if core is None:
            core = -1
        BifrostObject.__init__(
            self, _bf.bfUdpCaptureCreate, _bf.bfPacketCaptureDestroy,
            fmt.encode(), sock.fileno(), ring.obj, nsrc, src0,
            max_payload_size, buffer_ntime, slot_ntime,
            sequence_callback.obj, core)

class UDPSniffer(_CaptureBase):
    def __init__(self, fmt: str, sock: UDPSocket, ring: Union[Ring,Ring2],
                 nsrc: int, src0: int, max_payload_size: int, buffer_ntime: int,
                 slot_ntime: int, sequence_callback: PacketCaptureCallback,
                 core: Optional[int]=None):
        nsrc = self._flatten_value(nsrc)
        if core is None:
            core = -1
        BifrostObject.__init__(
            self, _bf.bfUdpSnifferCreate, _bf.bfPacketCaptureDestroy,
            fmt.encode(), sock.fileno(), ring.obj, nsrc, src0,
            max_payload_size, buffer_ntime, slot_ntime,
            sequence_callback.obj, core)

class UDPVerbsCapture(_CaptureBase):
    def __init__(self, fmt: str, sock: UDPSocket, ring: Union[Ring,Ring2],
                 nsrc: int, src0: int, max_payload_size: int, buffer_ntime: int,
                 slot_ntime: int, sequence_callback: PacketCaptureCallback,
                 core: Optional[int]=None):
        if core is None:
            core = -1
        BifrostObject.__init__(
            self, _bf.bfUdpVerbsCaptureCreate, _bf.bfPacketCaptureDestroy,
            fmt.encode(), sock.fileno(), ring.obj, nsrc, src0,
            max_payload_size, buffer_ntime, slot_ntime,
            sequence_callback.obj, core)

class DiskReader(_CaptureBase):
    def __init__(self, fmt: str, fh: IOBase, ring: Union[Ring,Ring2],
                 nsrc: int, src0: int, buffer_nframe: int, slot_nframe: int,
                 sequence_callback: PacketCaptureCallback,
                 core: Optional[int]=None):
        nsrc = self._flatten_value(nsrc)
        if core is None:
            core = -1
        BifrostObject.__init__(
            self, _bf.bfDiskReaderCreate, _bf.bfPacketCaptureDestroy,
            fmt.encode(), fh.fileno(), ring.obj, nsrc, src0,
            buffer_nframe, slot_nframe,
            sequence_callback.obj, core)
        # Make sure we start in the same place in the file
        self.seek(fh.tell(), _bf.BF_WHENCE_SET)
    def seek(self, offset: int, whence: _th.BFwhence_enum=_bf.BF_WHENCE_CUR):
        position = ctypes.c_ulong(0)
        _check(_bf.bfPacketCaptureSeek(self.obj, offset, whence, position))
        return position.value
    def tell(self) -> int:
        position = ctypes.c_ulong(0)
        _check(_bf.bfPacketCaptureTell(self.obj, position))
        return position.value
