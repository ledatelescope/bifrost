# -*- coding: utf-8 -*-
from libbifrost import _bf, _check, _get, _string2space, _space2string

import ctypes
import numpy as np

def _packet2pointer(packet):
	buf = ctypes.create_string_buffer(packet)
	len = ctype.sizeof(buf)
	return ctypes.pointer(buf), len


def _packets2pointer(packets):
	count = len(packets)
	buf = ctypes.create_string_buffer("".join(packets))
	len = ctypes.sizeof(buf)/count
	return ctypes.pointer(buf), len, count


class UDPTransmit(object):
	def __init__(self, sock, core=-1):
		self.obj = None
		self.obj = _get(_bf.UdpTransmitCreate(fd=sock.fileno(),
		                                     core=core), retarg=0)
	def __del__(self):
		if hasattr(self, 'obj') and bool(self.obj):
			_bf.UdpTransmitDestroy(self.obj)
	def __enter__(self):
		return self
	def __exit__(self, type, value, tb):
		pass
	def send(self, packet):
		ptr, len = _packet2pointer(packet)
		return _get( _bf.UdpTransmitSend(self.obj, ptr, len) )
	def sendmany(self, packets):
		assert(type(packets) is list)
		ptr, len, count = _packets2pointer(packets)
		return _get( _bf.UdpTransmitSendMany(self.obj, ptr, len, count) )