
# Copyright (c) 2022-2024, The Bifrost Authors. All rights reserved.
# Copyright (c) 2022-2024, The University of New Mexico. All rights reserved.
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

from bifrost.libbifrost import _bf, _th, _check, _get, BifrostObject
from bifrost.ndarray import ndarray, asarray
from bifrost.proclog import ProcLog
import bifrost.affinity as cpu_affinity
from bifrost.udp_socket import UDPSocket
from bifrost.ring import Ring
from bifrost.ring2 import Ring as Ring2

import time
import ctypes

from typing import Optional, Tuple, Union

from bifrost import telemetry
telemetry.track_module()

class RdmaSender(BifrostObject):
    def __init__(self, sock: UDPSocket, message_size: int):
        BifrostObject.__init__(
            self, _bf.bfRdmaCreate, _bf.bfRdmaDestroy,
            sock.fileno(), message_size, 1)
    def send_header(self, time_tag: int, header: str, offset_from_head: int):
        header = header.encode()
        header_buf = ctypes.create_string_buffer(header)
        _check(_bf.bfRdmaSendHeader(self.obj,
                                    time_tag,
                                    len(header),
                                    ctypes.cast(header_buf, ctypes.c_void_p),
                                    offset_from_head))
    def send_span(self, span_data: ndarray):
        _check(_bf.bfRdmaSendSpan(self.obj,
                                  asarray(span_data).as_BFarray()))

class RdmaReceiver(BifrostObject):
    def __init__(self, sock: UDPSocket, message_size: int, buffer_factor: int=5):
        BifrostObject.__init__(
            self, _bf.bfRdmaCreate, _bf.bfRdmaDestroy,
            sock.fileno(), message_size, 0)
        self.message_size = message_size
        self.buffer_factor = buffer_factor
        
        self.time_tag = ctypes.c_ulong(0)
        self.header_size = ctypes.c_ulong(0)
        self.offset_from_head = ctypes.c_ulong(0)
        self.span_size = ctypes.c_ulong(0)
        self.contents_bufs = []
        for i in range(self.buffer_factor):
            contents_buf = ctypes.create_string_buffer(self.message_size)
            self.contents_bufs.append(contents_buf)
        self.index = 0
    def receive(self) -> Union[Tuple[int,int,str],ndarray]:
        contents_buf = self.contents_bufs[self.index]
        self.index += 1
        if self.index == self.buffer_factor:
            self.index = 0
            
        _check(_bf.bfRdmaReceive(self.obj,
                                 ctypes.POINTER(ctypes.c_ulong)(self.time_tag),
                                 ctypes.POINTER(ctypes.c_ulong)(self.header_size),
                                 ctypes.POINTER(ctypes.c_ulong)(self.offset_from_head),
                                 ctypes.POINTER(ctypes.c_ulong)(self.span_size),
                                 ctypes.addressof(contents_buf)))
        if self.header_size.value > 0:
            contents = ctypes.cast(contents_buf, ctypes.c_char_p)
            contents = contents.value            
            return self.time_tag.value, self.header_size.value, contents
        else:
            span_data = ndarray(shape=(self.span_size.value,), dtype='u8', buffer=ctypes.addressof(contents_buf))
            return span_data

class RingSender(object):
    def __init__(self, iring: Union[Ring,Ring2], sock: UDPSocket, gulp_size: int,
                 guarantee: bool=True, core: Optional[int]=None):
        self.iring = iring
        self.sock = sock
        self.gulp_size = gulp_size
        self.guarantee = guarantee
        self.core = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        
    def main(self):
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        sender = RdmaSender(self.sock, self.gulp_size)
        
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            sender.send_header(iseq.time_tag, len(ihdr), ihdr, 0)
            self.sequence_proclog.update(ihdr)
            
            prev_time = time.time()
            iseq_spans = iseq.read(self.gulp_size)
            for ispan in iseq_spans:
                if ispan.size < self.gulp_size:
                    continue
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                sender.send_span(ispan)
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})


class RingReceiver(object):
    def __init__(self, oring: Union[Ring,Ring2], sock: UDPSocket, gulp_size: int,
                 guarantee: bool=True, core: Optional[int]=None):
        self.oring = oring
        self.sock = sock
        self.gulp_size = gulp_size
        self.guarantee = guarantee
        self.core = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.out_proclog   = ProcLog(type(self).__name__+"/out")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.out_proclog.update(  {'nring':1, 'ring0':self.oring.name})
        
    def main(self):
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        receiver = RdmaReceiver(self.sock, self.gulp_size)
        
        with self.oring.begin_writing() as oring:
            prev_time = time.time()
            data = receiver.receive()
            while True:
                while not isinstance(data, tuple):
                    data = receiver.receive()
                time_tag, _, ihdr = data
                self.sequence_proclog.update(ihdr)
                
                ohdr = ihdr.copy()
                
                with oring.begin_sequence(time_tag=time_tag, header=ohdr) as oseq:
                    while True:
                        data = receiver.receive()
                        if isinstance(data, tuple):
                            break
                        curr_time = time.time()
                        acquire_time = curr_time - prev_time
                        prev_time = curr_time
                        
                        with oseq.reserve(self.gulp_size) as ospan:
                            curr_time = time.time()
                            reserve_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            odata[...] = data
                            
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,})
