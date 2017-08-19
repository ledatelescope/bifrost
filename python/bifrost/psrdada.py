
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

"""
This provides an interface for reading/writing PSRDADA ring buffers.

bifrost.libpsrdada_generated is generated at build time using ctypesgen.py

PSRDADA must be built as a shared library to use this. This can be accomplished
by adding the following lines to psrdada/configure.in:

#AC_DISABLE_SHARED
LT_INIT
lib_LTLIBRARIES = libpsrdada.la
libtest_la_LDFLAGS = -version-info 0:0:0
"""

from __future__ import absolute_import

from bifrost.pipeline import SourceBlock, SinkBlock
from bifrost.DataType import DataType
import bifrost.libpsrdada_generated as _dada
import numpy as np
from bifrost.ndarray import _address_as_buffer

from copy import deepcopy
import os
import ctypes

def get_pointer_value(ptr):
    return ctypes.c_void_p.from_buffer(ptr).value

class MultiLog(object):
    count = 0
    def __init__(self, name=None):
        if name is None:
            name = "MultiLog%i" % MultiLog.count
            MultiLog.count += 1
        self.obj = _dada.multilog_open(name, '\0')
    def __del__(self):
        _dada.multilog_close(self.obj)

class IpcBufBlock(object):
    def __init__(self, buf, mutable=False):
        self.buf = buf
        self.ptr, self.nbyte, self.block_id = self.buf.open()
        self.nbyte_commit = self.nbyte
        self.ptr = get_pointer_value(self.ptr)
        if self.ptr is not None:
            self.data = np.ndarray(
                shape=(self.nbyte,),
                buffer=_address_as_buffer(self.ptr, self.nbyte),
                dtype=np.uint8)
            self.data.flags['WRITEABLE'] = mutable
    def __del__(self):
        self.close()
    def commit(self, nbyte=None):
        if nbyte is None:
            nbyte = self.nbyte
        self.nbyte_commit = nbyte
    def close(self):
        if self.ptr is not None:
            self.buf.close(self.nbyte_commit)
            self.ptr = None
    def enable_eod(self):
        #print '>ipcbuf_enable_eod'
        if _dada.ipcbuf_enable_eod(self.buf.buf) < 0:
            raise IOError("Failed to enable EOD flag")
    def size_bytes(self):
        return self.nbyte
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.close()

class IpcBaseBuf(object):
    def __init__(self, ipcbuf, mutable=False):
        self.buf = ipcbuf
        self.mutable = mutable
    def size_bytes(self):
        return _dada.ipcbuf_get_bufsz(self.buf)
    def eod(self):
        #print '>ipcbuf_eod'
        return bool(_dada.ipcbuf_eod(self.buf))
    def reset(self):
        #print '>ipcbuf_reset'
        if _dada.ipcbuf_reset(self.buf) < 0:
            raise IOError("Failed to reset buffer")
    def __iter__(self):
        return self
    def next(self):
        block = IpcBufBlock(self, self.mutable)
        if block.nbyte > 0:
            return block
        else:
            del block
            self.reset()
            raise StopIteration()
    def open(self):
        raise NotImplementedError()
    def close(self):
        raise NotImplementedError()

class IpcBaseIO(IpcBaseBuf):
    def __init__(self, ipcio, mutable=False):
        ipcbuf = ctypes.cast(ipcio, ctypes.POINTER(_dada.ipcbuf_t))
        super(IpcBaseIO, self).__init__(ipcbuf, mutable)
        self.io = ipcio
    def stop(self):
        #print '>ipcio_stop'
        if _dada.ipcio_stop(self.io) < 0:
            raise IOError("Failed to write EOD marker to block")

class IpcReadHeaderBuf(IpcBaseBuf):
    def __init__(self, ipcbuf):
        super(IpcReadHeaderBuf, self).__init__(ipcbuf)
    def open(self):
        nbyte = ctypes.c_uint64()
        #print '>ipcbuf_get_next_read'
        ptr = _dada.ipcbuf_get_next_read(self.buf, nbyte)
        nbyte = nbyte.value
        block_id = 0
        return ptr, nbyte, block_id
    def close(self, nbyte):
        #print '>ipcbuf_mark_cleared'
        if _dada.ipcbuf_mark_cleared(self.buf) < 0:
            raise IOError("Failed to mark block as cleared")

class IpcWriteHeaderBuf(IpcBaseBuf):
    def __init__(self, ipcbuf):
        super(IpcWriteHeaderBuf, self).__init__(ipcbuf, mutable=True)
    def open(self):
        nbyte = self.size_bytes()
        #print '>ipcbuf_get_next_write'
        ptr = _dada.ipcbuf_get_next_write(self.buf)
        block_id = 0
        return ptr, nbyte, block_id
    def close(self, nbyte):
        #print '>ipcbuf_mark_filled'
        if _dada.ipcbuf_mark_filled(self.buf, nbyte) < 0:
            raise IOError("Failed to mark block as filled")

class IpcReadDataBuf(IpcBaseIO):
    def __init__(self, ipcio):
        super(IpcReadDataBuf, self).__init__(ipcio)
    def open(self):
        nbyte    = ctypes.c_uint64()
        block_id = ctypes.c_uint64()
        #print '>ipcio_open_block_read'
        ptr = _dada.ipcio_open_block_read(self.io, nbyte, block_id)
        nbyte = nbyte.value
        block_id = block_id.value
        #print 'block_id =', block_id
        return ptr, nbyte, block_id
    def close(self, nbyte):
        #print '>ipcio_close_block_read(nbyte=%i)' % nbyte
        if _dada.ipcio_close_block_read(self.io, nbyte) < 0:
            raise IOError("Failed to close block for reading")

class IpcWriteDataBuf(IpcBaseIO):
    def __init__(self, ipcio):
        super(IpcWriteDataBuf, self).__init__(ipcio, mutable=True)
        self.nbyte_commit = 0 # Default to committing nothing
    def open(self):
        nbyte = self.size_bytes()
        block_id = ctypes.c_uint64()
        #print '>ipcio_open_block_write'
        ptr = _dada.ipcio_open_block_write(self.io, block_id)
        block_id = block_id.value
        #print 'block_id =', block_id
        return ptr, nbyte, block_id
    def close(self, nbyte):
        #print '>ipcio_close_block_write(nbyte=%i)' % nbyte
        if _dada.ipcio_close_block_write(self.io, nbyte) < 0:
            raise IOError("Failed to close block for writing")

class Hdu(object):
    def __init__(self):
        self._dada = _dada
        self.log = MultiLog()
        self.hdu = _dada.dada_hdu_create(self.log.obj)
        self.connected = False
    def __del__(self):
        self.disconnect()
        _dada.dada_hdu_destroy(self.hdu)
    def _connect(self, buffer_key=0xDADA):
        self.buffer_key = buffer_key
        _dada.dada_hdu_set_key(self.hdu, self.buffer_key)
        if _dada.dada_hdu_connect(self.hdu) < 0:
            raise IOError("Could not connect to buffer '%x'" % self.buffer_key)
    def _disconnect(self):
        if _dada.dada_hdu_disconnect(self.hdu) < 0:
            raise IOError("Could not disconnect from buffer '%x'" % self.buffer_key)
    def _lock(self, mode):
        self.mode = mode
        if mode == 'read':
            if _dada.dada_hdu_lock_read(self.hdu) < 0:
                raise IOError("Could not lock buffer '%x' for reading" % self.buffer_key)
        else:
            if _dada.dada_hdu_lock_write(self.hdu) < 0:
                raise IOError("Could not lock buffer '%x' for writing" % self.buffer_key)
    def _unlock(self):
        if self.mode == 'read':
            if _dada.dada_hdu_unlock_read(self.hdu) < 0:
                raise IOError("Could not unlock buffer '%x' for reading" % self.buffer_key)
        else:
            if _dada.dada_hdu_unlock_write(self.hdu) < 0:
                raise IOError("Could not unlock buffer '%x' for writing" % self.buffer_key)
    def relock(self):
        self._unlock()
        self._lock(self.mode)
    def open_HACK(self):
        if _dada.ipcio_open(self.data_block.io, 'w') < 0:
            raise IOError("ipcio_open failed")
    def connect_read(self, buffer_key=0xDADA):
        self._connect(buffer_key)
        self._lock('read')
        self.header_block = IpcReadHeaderBuf(self.hdu.contents.header_block)
        self.data_block   = IpcReadDataBuf(self.hdu.contents.data_block)
        self.connected = True
    def connect_write(self, buffer_key=0xDADA):
        self._connect(buffer_key)
        self._lock('write')
        self.header_block = IpcWriteHeaderBuf(self.hdu.contents.header_block)
        self.data_block   = IpcWriteDataBuf(self.hdu.contents.data_block)
        self.connected = True
    def disconnect(self):
        if self.connected:
            self._unlock()
            self._disconnect()
            self.connected = False
