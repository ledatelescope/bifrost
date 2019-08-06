
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

from __future__ import absolute_import

from bifrost.pipeline import SourceBlock, SinkBlock
from bifrost.Space import Space
from bifrost.psrdada import Hdu
from bifrost.libbifrost import _bf, _check
import bifrost.ndarray

from copy import deepcopy
import os

# TODO: Move to memory.py?
def _get_space(arr):
    if isinstance(arr, bifrost.ndarray):
        return arr.bf.space
    else:
        return 'system'

class PsrDadaBufferReader(object):
    def __init__(self, hdu, hdr):
        self.hdu = hdu
        self.header = hdr
        self._open_next_block()
    def _open_next_block(self):
        self.block = next(self.hdu.data_block)
        self.nbyte = self.block.size_bytes()
        self.byte0 = 0
    def readinto(self, buf):
        dst_space = Space(_get_space(buf)).as_BFspace()
        byte0 = 0
        nbyte = buf.nbytes
        nbyte_copy = min(nbyte - byte0, self.nbyte - self.byte0)
        while nbyte_copy:
            _check(_bf.bfMemcpy(buf.ctypes.data + byte0, dst_space,
                                self.block.ptr + self.byte0, _bf.BF_SPACE_SYSTEM,
                                nbyte_copy))
            byte0      += nbyte_copy
            self.byte0 += nbyte_copy
            nbyte_copy = min(nbyte - byte0, self.nbyte - self.byte0)
            if self.nbyte - self.byte0 == 0:
                self.block.close()
                try:
                    self._open_next_block()
                except StopIteration:
                    break
        return byte0
    def close(self):
        self.block.close()
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.close()

def psrdada_read_sequence_iterator(buffer_key):
    hdu = Hdu()
    hdu.connect_read(buffer_key)
    for hdr in hdu.header_block:
        with hdr:
            hdr_string = hdr.data.tostring()
        yield hdu, hdr_string
    hdu.disconnect()

def _cast_to_type(string):
    try: return int(string)
    except ValueError: pass
    try: return float(string)
    except ValueError: pass
    return string
def parse_dada_header(headerstr, cast_types=True):
    headerstr = headerstr[:headerstr.find('\0')]
    header = {}
    for line in headerstr.split('\n'):
        try:
            key, value = line.split(None, 1)
        except ValueError:
            break
        key = key.strip()
        value = value.strip()
        if cast_types:
            value = _cast_to_type(value)
        header[key] = value
    return header

class PsrDadaSourceBlock(SourceBlock):
    def __init__(self, buffer_key, header_callback, gulp_nframe, space=None,
                 *args, **kwargs):
        buffer_iterator = psrdada_read_sequence_iterator(buffer_key)
        super(PsrDadaSourceBlock, self).__init__(buffer_iterator, gulp_nframe,
                                                 space, *args, **kwargs)
        self.header_callback = header_callback
    def create_reader(self, hdu_hdr):
        hdu, hdr = hdu_hdr
        return PsrDadaBufferReader(hdu, hdr)
    def on_sequence(self, reader, hdu_hdr):
        ihdr_dict = parse_dada_header(reader.header)
        return [self.header_callback(ihdr_dict)]
    def on_data(self, reader, ospans):
        ospan = ospans[0]
        nbyte = reader.readinto(ospan.data)
        if nbyte % ospan.frame_nbyte != 0:
            raise IOError("Block data is truncated")
        nframe = nbyte // ospan.frame_nbyte
        return [nframe]

def read_psrdada_buffer(buffer_key, header_callback, gulp_nframe, space=None,
                        *args, **kwargs):
    """Read data from a PSRDADA ring buffer.

    Args:
        buffer_key (int): Integer key of the shared memory buffer to read
          (e.g., 0xDADA).
        header_callback (func): A function f(psrdata_header_dict) -> bifrost_header_dict.
        gulp_nframe (int): No. frames to process at a time.
        space (string): The output memory space (all Bifrost spaces are supported).
        *args: Arguments to ``bifrost.pipeline.SourceBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.SourceBlock``.

    **Tensor semantics**::

        Output: metadata defined via header_callback, space = ANY

    Returns:
        PsrDadaSourceBlock: A new block instance.

    Note:
        PSRDADA must be built and installed as a shared library (libpsrdada.so)
        to use this. This can be accomplished by adding the following lines to
        psrdada/configure.in in the psrdada repository::

            #AC_DISABLE_SHARED
            LT_INIT
            lib_LTLIBRARIES = libpsrdada.la
            libtest_la_LDFLAGS = -version-info 0:0:0

    References:
        http://psrdada.sourceforge.net/
    """
    return PsrDadaSourceBlock(buffer_key, header_callback, gulp_nframe, space,
                              *args, **kwargs)
