
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

from libbifrost import _bf, _check, _get, _string2space, _space2string

import ctypes
import numpy as np

class UDPSocket(object):
	def __init__(self):
		self.obj = _get(_bf.UdpSocketCreate(), retarg=0)
	def __del__(self):
		if hasattr(self, 'obj') and bool(self.obj):
			_bf.UdpSocketDestroy(self.obj)
	def bind(self, local_addr):
		_check( _bf.UdpSocketBind(self.obj, local_addr.obj) )
	def connect(self, remote_addr):
		_check( _bf.UdpSocketConnect(self.obj, remote_addr.obj) )
	def shutdown(self):
		_check( _bf.UdpSocketShutdown(self.obj) )
	def close(self):
		_check( _bf.UdpSocketClose(self.obj) )
	@property
	def mtu(self):
		return _get(_bf.UdpSocketGetMTU(self.obj))
	def fileno(self):
		return _get(_bf.UdpSocketGetFD(self.obj))
	@property
	def timeout(self):
		return _get( _bf.UdpSocketGetTimeout(self.obj) )
	@timeout.setter
	def timeout(self, secs):
		_check( _bf.UdpSocketSetTimeout(self.obj, secs) )
