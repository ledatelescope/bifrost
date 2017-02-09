# -*- coding: utf-8 -*-
from libbifrost import _bf, _check, _get

import ctypes
import numpy as np

def _packet2pointer(packet):
	buf = ctypes.c_char_p(packet)
	siz = ctypes.c_uint( len(packet) )
	return buf, siz


def _packets2pointer(packets):
	count = ctypes.c_uint( len(packets) )
	buf = ctypes.c_char_p("".join(packets))
	siz = ctypes.c_uint( len(packets[0]) )
	return buf, siz, count


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
		ptr, siz = _packet2pointer(packet)
		return _get( _bf.UdpTransmitSend(self.obj, ptr, siz) )
	def sendmany(self, packets):
		assert(type(packets) is list)
		ptr, siz, count = _packets2pointer(packets)
		return _get( _bf.UdpTransmitSendMany(self.obj, ptr, siz, count) )