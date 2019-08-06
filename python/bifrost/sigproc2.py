
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
telescope_id:  0 (FAKE)
machine_id:    0 (FAKE)
data_type:     2 # Time-series data
rawdatafile:   <delete>
source_name:   <observer-specified>
barycentric:   0
pulsarcentric: <delete>
az_start:      <delete> or <observer-specified>
za_start:      <delete> or <observer-specified>
src_raj:       <observer-specified> or <delete>
src_dej:       <observer-specified> or <delete>
tstart:        MJD of first sample
tsamp:         (secs) E.g., 0.5/(2400*24kHz=57.6MHz)=8.68055556ns
nbits:         8
nsamples:      No. time samples in file ("rarely used any more")
fch1:          58.776 MHz (center frequency)
foff:           MHz
nchans:        1
nifs:          2 (pols)
refdm:         0.0 [pc/cm^3]
period:        <delete>
data:          [time][pol][nbit] (General case: [time][if/pol][chan][nbit])
"""

# See here for details of the different data formats:
#   https://github.com/SixByNine/sigproc

import struct
import numpy as np
from collections import defaultdict

_string_values = ['source_name',
                  'rawdatafile']
_double_values = ['az_start',
                  'za_start',
                  'src_raj',
                  'src_dej',
                  'tstart',
                  'tsamp',
                  'period',
                  'fch1',
                  'foff',
                  'refdm']
_integer_values = ['nchans',
                   'telescope_id',
                   'machine_id',
                   'data_type',
                   'ibeam',
                   'nbeams',
                   'nbits',
                   'barycentric',
                   'pulsarcentric',
                   'nbins',
                   'nsamples',
                   'nifs',
                   'npuls']
_character_values = ['signed']

# Note: 1, 2 and 6 are basically the same thing:
# nchan=1 &&  refdm => time series
# nchan>1 &&  refdm => dedispersed time series
# nchan>1 && !refdm => filterbank
#   So, only 'pulse profile' requires special handling (and note that it's not a streaming format)
_data_types = defaultdict(lambda: 'unknown',
                          {0: 'raw data',
                           1: 'filterbank',            # [time,pol,chan]
                           2: 'time series',           # [time,pol] (refdm)
                           3: 'pulse profile',         # [pol,chan,bin] (nbins, period, optional npuls)
                           4: 'amplitude spectrum',    # ???
                           5: 'complex spectrum',      # ???
                           6: 'dedispersed subbands'}) # [time,pol,subband] (refdm; basically part-way between filterbank and time-series)
_telescopes = defaultdict(lambda: 'unknown',
                          {0:  'Fake',
                           1:  'Arecibo',
                           2:  'Ooty',
                           3:  'Nancay',
                           4:  'Parkes',  # TODO: Should be 7?
                           5:  'Jodrell', # 'Lovell',
                           6:  'GBT',
                           7:  'GMRT',
                           8:  'Effelsberg',
                           9:  'ATA',
                           10: 'UTR-2',
                           11: 'LOFAR',
                           52: 'LWA-OV',
                           53: 'LWA-SV'})
_machines   = defaultdict(lambda: 'unknown',
                          {0:  'FAKE',
                           1:  'PSPM',
                           2:  'WAPP',
                           3:  'AOFTM',
                           4:  'BPP', # aka BCPM1
                           5:  'OOTY', # TODO: Should be 3?
                           6:  'SCAMP',
                           7:  'GMRTFB', # aka GBT Pulsar Spigot, SPIGOT
                           8:  'PULSAR2000',
                           11: 'BG/P',
                           12: "PDEV",
                           20: 'GUPPI',
                           52: 'LWA-DP',
                           53: 'LWA-ADP'})

def id2telescope(id_):
    return _telescopes[id_]
def telescope2id(name):
    # TODO: Would be better to use a pre-made reverse lookup dict
    return _telescopes.keys()[_telescopes.values().index(name)]
def id2machine(id_):
    return _machines[id_]
def machine2id(name):
    # TODO: Would be better to use a pre-made reverse lookup dict
    return _machines.keys()[_machines.values().index(name)]

def _header_write_string(f, key):
    f.write(struct.pack('=i', len(key)))
    f.write(key)
def _header_write(f, key, value, fmt=None):
    if fmt is not None:
        pass
    elif isinstance(value, int):
        fmt = '=i'
    elif isinstance(value, float):
        fmt = '=d'
    #elif key == 'signed':
    #    fmt = '=b'
    else:
        raise TypeError("Invalid value type")
    _header_write_string(f, key)
    f.write(struct.pack(fmt, value))

def _header_read(f):
    length = struct.unpack('=i', f.read(4))[0]
    if length < 0 or length >= 80:
        return None
    s = f.read(length)
    return s

def write_header(hdr, f):
    _header_write_string(f, "HEADER_START")
    for key, val in hdr.items():
        if val is None:
            # Do not write keys with no value
            continue
        if key in _string_values:
            _header_write_string(f, key)
            _header_write_string(f, val)
        elif key in _double_values:
            _header_write(f, key, float(val))
        elif key in _integer_values:
            _header_write(f, key, int(val))
        elif key in _character_values:
            _header_write(f, key, int(val), fmt='=b')
        else:
            #raise KeyError("Unknown sigproc header key: %s"%key)
            print "WARNING: Unknown sigproc header key: %s" % key
    _header_write_string(f, "HEADER_END")

def _read_header(f):
    if _header_read(f) != "HEADER_START":
        #f.seek(0)
        raise ValueError("Missing HEADER_START")
    expecting = None
    header = {}
    while True:
        key = _header_read(f)
        if key is None:
            raise ValueError("Failed to parse header")
        elif key == 'HEADER_END':
            break
        elif key in _string_values:
            expecting = key
        elif key in _double_values:
            header[key] = struct.unpack('=d', f.read(8))[0]
        elif key in _integer_values:
            header[key] = struct.unpack('=i', f.read(4))[0]
        elif key in _character_values:
            header[key] = struct.unpack('=b', f.read(1))[0]
        elif expecting is not None:
            header[expecting] = key
            expecting = None
        else:
            print "WARNING: Unknown header key", key
    if 'nchans' not in header:
        header['nchans'] = 1
    header['header_size'] = f.tell()
    #frame_bits = header['nifs'] * header['nchans'] * header['nbits']
    #if 'nsamples' not in header or header['nsamples'] == 0:
    #    f.seek(0, 2) # Seek to end of file
    #    header['nsamples'] = (f.tell() - header['header_size'])*8 / frame_bits
    #    f.seek(header['header_size'], 0) # Seek back to end of header
    return header

# TODO: Move this elsewhere?
def unpack(data, nbit):
    if nbit > 8:
        raise ValueError("unpack: nbit must be <= 8")
    if 8 % nbit != 0:
        raise ValueError("unpack: nbit must divide into 8")
    if data.dtype not in (np.uint8, np.int8):
        raise TypeError("unpack: dtype must be 8-bit")
    if nbit == 8:
        return data
    elif nbit == 4:
        # Note: This technique assumes least-significant-bit-first ordering
        x = data.astype(np.int16)
        x = (x | (x <<  4)) & 0x0F0F
        x = x << 4 # Shift into high bits to induce sign-extension
        return x.view(data.dtype) >> 4
    elif nbit == 2:
        x = data.astype(np.int32)
        x = (x | (x << 12)) & 0x000F000F
        x = (x | (x <<  6)) & 0x03030303
        x = x << 6 # Shift into high bits to induce sign-extension
        return x.view(data.dtype) >> 6
    elif nbit == 1:
        x = data.astype(np.int64)
        x = (x | (x << 28)) & 0x0000000F0000000F
        x = (x | (x << 14)) & 0x0003000300030003
        x = (x | (x <<  7)) & 0x0101010101010101
        x = x << 7 # Shift into high bits to induce sign-extension
        return x.view(data.dtype) >> 7
    else:
        raise ValueError("unpack: unexpected nbit! (%i)" % nbit)

# TODO: Add support for writing
#       Add support for data_type != filterbank
class SigprocFile(object):
    def __init__(self, filename=None):
        if filename is not None:
            self.open(filename)
    def open(self, filename):
        # Note: If nbit < 8, pack_factor = 8 / nbit and the last dimension
        #         is divided by pack_factor, with dtype set to uint8.
        self.f = open(filename, 'rb')
        self.header = _read_header(self.f)
        self.header_size = self.header['header_size']
        self.frame_shape = (self.header['nifs'], self.header['nchans'])
        self.nbit = self.header['nbits']
        if 'signed' not in self.header:
            self.header['signed'] = False
        self.signed = bool(self.header['signed'])
        if self.nbit >= 8:
            if self.signed:
                self.dtype  = { 8: np.int8,
                               16: np.int16,
                               32: np.float32,
                               64: np.float64}[self.nbit]
            else:
                self.dtype  = { 8: np.uint8,
                               16: np.uint16,
                               32: np.float32,
                               64: np.float64}[self.nbit]
        else:
            # **TODO: This is broken when nchan < pack_factor
            #           A proper solution to this would be to track read/write
            #             offsets in units of bits instead of bytes, but this
            #             would require changing everything in ring too.
            #             An alternative is to only allow reads/writes in
            #               gulps that correspond to a whole number of bytes.
            #               E.g., nchan=1,nbit=4 => read/write size must be a
            #                 multiple of 2 frames.
            #self.dtype = np.int8 if self.signed else np.uint8
            #pack_factor = 8 / self.nbit
            #self.frame_shape = (self.frame_shape[0],
            #                    self.frame_shape[1]/pack_factor)
            ##self.frame_shape[-1] /= pack_factor
            self.dtype = None
        self.frame_size = self.frame_shape[0] * self.frame_shape[1]
        #self.frame_nbyte = self.frame_size*self.dtype().itemsize
        self.buf = np.empty(4096, np.uint8)
        self.frame_nbit  = self.frame_size * self.nbit
        self.frame_nbyte = self.frame_nbit // 8
        return self
    def close(self):
        self.f.close()
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.close()
    def seek(self, offset, whence=0):
        if whence == 0:
            offset += self.header_size
        self.f.seek(offset, whence)
    def bandwidth(self):
        return self.header['nchans'] * self.header['foff']
    def cfreq(self):
        return (self.header['fch1'] +
                0.5 * (self.header['nchans'] - 1) * self.header['foff'])
    def duration(self):
        return self.header['tsamp'] * self.nframe()
    def nframe(self):
        if 'nsamples' not in self.header or self.header['nsamples'] == 0:
            curpos = self.f.tell()
            self.f.seek(0, 2) # Seek to end of file
            frame_bits = self.header['nifs'] * self.header['nchans'] * self.header['nbits']
            nframe = ((self.f.tell() - self.header['header_size']) *
                      8 / frame_bits)
            self.header['nsamples'] = nframe
            self.f.seek(curpos, 0) # Seek back to where we were
        return self.header['nsamples']
    def read(self, nframe_or_start, end=None):
        if end is not None:
            start = nframe_or_start or 0
            if start * self.frame_size * self.nbit % 8 != 0:
                raise ValueError("Start index must be aligned with byte boundary " +
                                 "(idx=%i, nbit=%i)" % (start, self.nbit))
            self.seek(start * self.frame_size * self.nbit // 8)
            if end == -1:
                end = self.nframe()
            nframe = end - start
        else:
            nframe = nframe_or_start
        if self.nbit < 8:
            if nframe * self.frame_size * self.nbit % 8 != 0:
                raise ValueError("No. frames must correspond to whole number of bytes " +
                                 "(idx=%i, nbit=%i)" % (nframe, self.nbit))
            #data = np.fromfile(self.f, np.uint8,
            #                   nframe * self.frame_size * self.nbit // 8)
            #requested_nbyte = nframe * self.frame_nbyte
            requested_nbyte = nframe * self.frame_nbyte * self.nbit // 8
            if self.buf.nbytes != requested_nbyte:
                self.buf.resize(requested_nbyte)
            nbyte = self.f.readinto(self.buf)
            if nbyte * 8 % self.frame_nbit != 0:
                raise IOError("File read returned incomplete frame (truncated file?)")
            if nbyte < self.buf.nbytes:
                self.buf.resize(nbyte)
            nframe = nbyte * 8 // (self.frame_size * self.nbit)
            data = self.buf
            data = unpack(data, self.nbit)
            data = data.reshape((nframe,) + self.frame_shape)
        else:
            data = np.fromfile(self.f, self.dtype, nframe * self.frame_size)
            if data.size % self.frame_size != 0:
                raise IOError("File read returned incomplete frame (truncated file?)")
            nframe = data.size // self.frame_size
        data = data.reshape((nframe,) + self.frame_shape)
        return data
    def readinto(self, buf):
        """Fills buf with raw bytes straight from the file"""
        return self.f.readinto(buf)
    def __str__(self):
        hmod = self.header.copy()
        d = hmod['data_type']
        hmod['data_type'] = "%i (%s)" % (d, _data_types[d])
        t = hmod['telescope_id']
        hmod['telescope_id'] = "%i (%s)" % (t, _telescopes[t])
        m = hmod['machine_id']
        hmod['machine_id']   = "%i (%s)" % (m, _machines[m])
        return '\n'.join(['% 16s: %s' % (key, val)
                          for (key, val) in hmod.items()])
    def __getitem__(self, key):
        if isinstance(key, type("")): # Header key lookup
            return self.header[key]
        elif isinstance(key, int): # Extract one time slice
            return self.read(key, key + 1)[0]
        elif isinstance(key, slice): # 1D slice
            start = key.start if key.start is not None else  0
            stop  = key.stop  if key.stop  is not None else -1
            data = self.read(start, stop)
            #data = self.read(stop) if start == 0 else \
            #       self.read(start, key.stop)
            return data[::key.step]
        elif isinstance(key, tuple): # ND key
            raise NotImplementedError
