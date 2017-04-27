
# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

from libbifrost import _bf, _check, _get, _fast_get, _string2space, _space2string

import ctypes
import numpy as np

class UDPCapture(object):
	def __init__(self, fmt, sock, ring, nsrc, src0, max_payload_size,
	             buffer_ntime, slot_ntime, sequence_callback, core=None):
		self.obj = None
		if core is None:
			core = -1
		self.obj = _get(_bf.UdpCaptureCreate(format=fmt,
		                                     fd=sock.fileno(),
		                                     ring=ring.obj,
		                                     nsrc=nsrc,
		                                     src0=src0,
		                                     max_payload_size=max_payload_size,
		                                     buffer_ntime=buffer_ntime,
		                                     slot_ntime=slot_ntime,
		                                     sequence_callback=sequence_callback,
		                                     core=core), retarg=0)
	def __del__(self):
		if hasattr(self, 'obj') and bool(self.obj):
			_bf.UdpCaptureDestroy(self.obj)
	def __enter__(self):
		return self
	def __exit__(self, type, value, tb):
		self.end()
	def recv(self):
		return _fast_get(_bf.UdpCaptureRecv, self.obj)
	def flush(self):
		_check( _bf.UdpCaptureFlush(self.obj) )
	def end(self):
		_check( _bf.UdpCaptureEnd(self.obj) )
