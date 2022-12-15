
# Copyright (c) 2016-2021, The Bifrost Authors. All rights reserved.
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

import numpy as np
from datetime import datetime
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
        self.block = None
    def _open_next_block(self):
        self.block = next(self.hdu.data_block)
        self.nbyte = self.block.size_bytes()
        self.byte0 = 0
    def readinto(self, buf):
        if self.block is None:
            self._open_next_block()
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
        if self.block:
            self.block.close()
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.close()

def psrdada_read_sequence_iterator(buffer_key, single=False):
    hdu = Hdu()
    hdu.connect_read(buffer_key)
    for hdr in hdu.header_block:
        with hdr:
            hdr_string = hdr.data.tostring()
        yield hdu, hdr_string
        if single:
            hdu.disconnect()
            break
    hdu.disconnect()

def _cast_to_type(string):
    try: return int(string)
    except ValueError: pass
    try: return float(string)
    except ValueError: pass
    return string

def _cast_to_string(unknown):
    if type(unknown) is bytes:
        return unknown.decode('utf-8')
    elif type(unknown) is str:
        return unknown

def parse_dada_header(headerstr, cast_types=True):
    headerstr = _cast_to_string(headerstr)
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
                 single=False, *args, **kwargs):
        buffer_iterator = psrdada_read_sequence_iterator(buffer_key, single)
        super(PsrDadaSourceBlock, self).__init__(buffer_iterator, gulp_nframe,
                                                 space, *args, **kwargs)
        self.header_callback = header_callback
    def create_reader(self, hdu_hdr):
        hdu, hdr = hdu_hdr
        return PsrDadaBufferReader(hdu, hdr)
    def on_sequence(self, reader, hdu_hdr):
        print("PsrDadaSourceBlock::on_sequence")
        ihdr_dict = parse_dada_header(reader.header)
        return [self.header_callback(ihdr_dict)]
    def on_data(self, reader, ospans):
        ospan = ospans[0]
        nbyte = reader.readinto(ospan.data)
        if nbyte % ospan.frame_nbyte != 0:
            raise IOError("Block data is truncated")
        nframe = nbyte // ospan.frame_nbyte
        return [nframe]

def _keyval_to_dadastr(key, val):
     """ Convert key: value pair into a DADA string """
     return "{key:20s} {val}\n".format(key=key.upper(), val=val)


def _extract_tensor_scale(key, tensor):
    try:
        idx = tensor['labels'].index(key)
        return tensor['scales'][idx]
    except ValueError as e:
        return [0, 0]


def _extract_tensor_shape(key, tensor):
    try:
        idx = tensor['labels'].index(key)
        return tensor['shape'][idx]
    except ValueError as e:
        return 1

     
def generate_dada_header(hdr_dict, hdrlen=4096):
    """ Generate DADA header from header dict 
    
    Args:
        hdr_dict (dict): Header dictionary of key, value pairs
        hdrlen (int): Size of header, default 4096
    
    Returns:
        s (str): DADA header string with padding to hdrlen
    """
    s = "HDR_VERSION         1.0\n"
    s+= "HDR_SIZE            %i\n" % hdrlen
    keys_to_skip = ('HDR_VERSION', 'HDR_SIZE')

    # update parameters from bifrost tensor
    if '_tensor' in hdr_dict.keys():
        print(hdr_dict['_tensor'])
        dtype = hdr_dict['_tensor']['dtype']
        dtype_vals = {
            'cf32': { 'NBIT': '32', 'NDIM': '2' },
            'f32': { 'NBIT': '32', 'NDIM': '1' },
            'ci8': { 'NBIT': '8', 'NDIM': '2' },
            'i8': { 'NBIT': '8', 'NDIM': '1' } }
        if dtype in dtype_vals.keys():
            hdr_dict['NBIT'] = dtype_vals[dtype]['NBIT']
            hdr_dict['NDIM'] = dtype_vals[dtype]['NDIM']

        hdr_dict['NBEAM'] =  _extract_tensor_shape("beam", hdr_dict['_tensor'])
        hdr_dict['NANT'] =  _extract_tensor_shape("station", hdr_dict['_tensor'])
        hdr_dict['NPOL'] = _extract_tensor_shape("pol", hdr_dict['_tensor'])
        hdr_dict['NCHAN'] = _extract_tensor_shape("freq", hdr_dict['_tensor'])

        f0 = _extract_tensor_scale("freq", hdr_dict['_tensor'])[0]
        chan_bw = _extract_tensor_scale("freq", hdr_dict['_tensor'])[1]
        bw = chan_bw * int(hdr_dict['NCHAN'])
        hdr_dict['BW'] = bw
        hdr_dict['FREQ'] = f0 + bw / 2

        # print(hdr_dict['_tensor'])
        ts = _extract_tensor_scale("time", hdr_dict['_tensor'])[0]

        tsamp = _extract_tensor_scale("fine_time", hdr_dict['_tensor'])[1]
        if tsamp == 0:
            tsamp = _extract_tensor_scale("time", hdr_dict['_tensor'])[1]
            if tsamp == 0:
                print("TSAMP was 0, changing to 10.24 us")
                tsamp = 0.00001024

        hdr_dict['TSAMP'] = tsamp * 1e6
        ts_integer = int(ts)
        hdr_dict['UTC_START'] = datetime.utcfromtimestamp(ts_integer).strftime("%Y-%m-%d-%H:%M:%S")

        fine_time = _extract_tensor_shape("fine_time", hdr_dict['_tensor'])

        bits_per_sample = int(hdr_dict['NBEAM']) * \
                          int(hdr_dict['NANT']) * \
                          int(hdr_dict['NPOL']) * \
                          int(hdr_dict['NBIT']) * \
                          int(hdr_dict['NDIM']) * \
                          int(hdr_dict['NCHAN'])

        resolution = (bits_per_sample * fine_time) / 8
        hdr_dict['RESOLUTION'] = resolution
   
        bytes_per_second = int((bits_per_sample / 8) / tsamp)
        hdr_dict['BYTES_PER_SECOND'] = bytes_per_second

        hdr_dict['FILE_SIZE'] = bytes_per_second * 8

        if hdr_dict['_tensor']['labels'] == ['time', 'beam', 'freq', 'fine_time']:
            hdr_dict['ORDER'] = 'SFT'

    for key, val in hdr_dict.items():
        if key not in keys_to_skip:
            if isinstance(val, (str, float, int)):
                s += _keyval_to_dadastr(key, val)
    s_padding = "\x00"
    if len(s) > hdrlen:
        raise RuntimeError("Header is too large for HDR_SIZE! Increase hdrlen")
    n_pad = hdrlen - len(s)
    return s + s_padding * n_pad

class PsrDadaSinkBlock(SinkBlock):
    def __init__(self, iring, buffer_key, gulp_nframe, space=None, *args, **kwargs):
        super(PsrDadaSinkBlock, self).__init__(iring, gulp_nframe, *args, **kwargs)
        self.hdu = Hdu()
        self.hdu.connect_write(buffer_key)
        self.keywords_to_add = {}
        self.keywords_to_sub = []
        self.keywords_to_change = {}

    def add_header_keywords(self, hdr_dict):
        """Add specified keywords to outgoing header dict"""
        for key, value in hdr_dict.items():
            self.keywords_to_add[key] = value

    def sub_header_keywords(self, hdr_dict):
        """Remove specified keywords from outgoing header dict"""
        for key in hdr_dict:
            self.keywords_to_sub.append(key)

    def remap_prefixed_keywords(self, prefix, suffixes):
        """Remap the keywords in suffixes, removing the prefix"""
        for suffix in suffixes:
            self.keywords_to_change[prefix + suffix] = suffix

    def on_sequence(self, iseq):
        print("PsrDadaSinkBlock::on_sequence")
        updated_header = iseq.header.copy()
        # rename some header keywords
        for key, value in self.keywords_to_change.items():
            try:
                self.keywords_to_add[value] = updated_header[key]
                self.keywords_to_sub.append(key)
            except KeyError:
                pass
        # insert the additional keywords
        updated_header.update(self.keywords_to_add)
        # remove the keywords
        for key in self.keywords_to_sub:
            try:
                del updated_header[key]
            except KeyError:
                pass
        dada_header_str = generate_dada_header(updated_header)
        dada_header_buf = next(self.hdu.header_block)            
        
        dada_header_buf.data[:] = np.fromstring(dada_header_str.encode('ascii'), dtype='uint8')
        dada_header_buf.close()
    
    def on_sequence_end(self, iseq):
        self.hdu.disconnect()

    def on_data(self, ispan):
        
        # TODO: Make this work in CUDA space 
        dada_blk = next(self.hdu.data_block)
        
        nbyte = ispan.data.nbytes
        _check(_bf.bfMemcpy(dada_blk.ptr, _bf.BF_SPACE_SYSTEM,
                            ispan.data.ctypes.data, _bf.BF_SPACE_SYSTEM,
                            nbyte))        
    
        #dada_blk.data[:] = ispan.data.view('u8')        
        dada_blk.close()

def read_psrdada_buffer(buffer_key, header_callback, gulp_nframe, space=None, single=False,
                        *args, **kwargs):
    """Read data from a PSRDADA ring buffer.

    Args:
        buffer_key (int): Integer key of the shared memory buffer to read
          (e.g., 0xDADA).
        header_callback (func): A function f(psrdata_header_dict) -> bifrost_header_dict.
        gulp_nframe (int): No. frames to process at a time.
        space (string): The output memory space (all Bifrost spaces are supported).
        single (bool): Only process a single data stream with the block
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
    return PsrDadaSourceBlock(buffer_key, header_callback, gulp_nframe, space, single,
                              *args, **kwargs)

def write_psrdada_buffer(iring, buffer_key, gulp_nframe, *args, **kwargs):
    """ Write into a PSRDADA ring buffer 
    
    Note:
        Initial version, currently only supports system space (not CUDA)
    """
    return PsrDadaSinkBlock(iring, buffer_key, gulp_nframe)

