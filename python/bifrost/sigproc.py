
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

#This program is for reading and writing sigproc filterbank files.

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

import struct
import numpy as np
from collections import defaultdict
import os

#header parameter names which precede strings
_STRING_VALUES = ['source_name',
                  'rawdatafile']
#header parameter names which precede doubles
_DOUBLE_VALUES = ['az_start',
                  'za_start',
                  'src_raj',
                  'src_dej',
                  'tstart',
                  'tsamp',
                  'period',
                  'fch1',
                  'foff',
                  'refdm']
#header parameter names which precede integers
_INTEGER_VALUES = ['nchans',
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
#this header parameter precedes a character
_CHARACTER_VALUES = ['signed']

#the data_type parameter names' translation
_DATA_TYPES = defaultdict(lambda: 'unknown',
                          {0: 'raw data',
                           1: 'filterbank',
                           2: 'time series',
                           3: 'pulse profile',
                           4: 'amplitude spectrum',
                           5: 'complex spectrum',
                           6: 'dedispersed subbands'})
#the telescope_id parameter names' translation
_TELESCOPES = defaultdict(lambda: 'unknown',
                          {0:  'Fake',
                           1:  'Arecibo',
                           2:  'Ooty',
                           3:  'Nancay',
                           4:  'Parkes',
                           5:  'Jodrell', # 'Lovell',
                           6:  'GBT',
                           7:  'GMRT',
                           8:  'Effelsberg',
                           52: 'LWA-OV',
                           53: 'LWA-SV'})
#the machine_id parameter names' translation
_MACHINES = defaultdict(lambda: 'unknown',
                        {0:  'FAKE',
                         1:  'PSPM',
                         2:  'WAPP',
                         3:  'AOFTM',
                         4:  'BPP',
                         5:  'OOTY',
                         6:  'SCAMP',
                         7:  'GMRTFB',
                         8:  'PULSAR2000',
                         52: 'LWA-DP',
                         53: 'LWA-ADP'})

def _header_write_string(file_object, key):
    """Writes a single key name to the header,
    which will be followed by the value"""
    file_object.write(struct.pack('=i', len(key)))
    file_object.write(key)

def _header_write_value(file_object, key, value):
    """Writes a single parameter value to the header"""
    if isinstance(value, int):
        fmt = '=i'
    elif isinstance(value, float):
        fmt = '=d'
    elif key == 'signed':
        fmt = '=b'
    else:
        raise TypeError("Invalid value type")
    _header_write_string(file_object, key)
    file_object.write(struct.pack(fmt, value))

def _header_read_one_parameter(file_object):
    """Reads a single key name from the header"""
    length = struct.unpack('=i', file_object.read(4))[0]
    if length <= 0 or length >= 80:
        return None
    return file_object.read(length)

def _write_header(hdr, file_object):
    """write the entire header to the current position of a file"""
    _header_write_string(file_object, "HEADER_START")
    for key, val in hdr.items():
        if key in _STRING_VALUES:
            _header_write_string(file_object, key)
            _header_write_string(file_object, val)
        elif key in _DOUBLE_VALUES:
            _header_write_value(file_object, key, float(val))
        elif key in _INTEGER_VALUES:
            _header_write_value(file_object, key, int(val))
        elif key == "header_size":
            pass
        else:
            #raise KeyError("Unknown sigproc header key: %s"%key)
            print "WARNING: Unknown sigproc header key: %s" % key
    _header_write_string(file_object, "HEADER_END")

def _read_header(file_object):
    """Get the entire header from a file, and return as dictionary"""
    file_object.seek(0)
    if _header_read_one_parameter(file_object) != "HEADER_START":
        file_object.seek(0)
        raise ValueError("Missing HEADER_START")
    expecting = None
    header = {}
    while True:
        key = _header_read_one_parameter(file_object)
        if key is None:
            raise ValueError("Failed to parse header")
        elif key == 'HEADER_END':
            break
        elif key in _STRING_VALUES:
            expecting = key
        elif key in _DOUBLE_VALUES:
            header[key] = struct.unpack('=d', file_object.read(8))[0]
        elif key in _INTEGER_VALUES:
            header[key] = struct.unpack('=i', file_object.read(4))[0]
        elif key in _CHARACTER_VALUES:
            header[key] = struct.unpack('=b', file_object.read(1))[0]
        elif expecting is not None:
            header[expecting] = key
            expecting = None
        else:
            print "WARNING: Unknown header key", key
    if 'nchans' not in header:
        header['nchans'] = 1
    header['header_size'] = file_object.tell()
    return header

def seek_to_data(file_object):
    """Go the the location in the file where the data begins"""
    file_object.seek(0)
    if _header_read_one_parameter(file_object) != "HEADER_START":
        file_object.seek(0)
        raise ValueError("Missing HEADER_START")
    expecting = None
    header = {}
    while True:
        key = _header_read_one_parameter(file_object)
        if key is None:
            raise ValueError("Failed to parse header")
        elif key == 'HEADER_END':
            break
        elif key in _STRING_VALUES:
            expecting = key
        elif key in _DOUBLE_VALUES:
            header[key] = struct.unpack('=d', file_object.read(8))[0]
        elif key in _INTEGER_VALUES:
            header[key] = struct.unpack('=i', file_object.read(4))[0]
        elif key in _CHARACTER_VALUES:
            header[key] = struct.unpack('=b', file_object.read(1))[0]
        elif expecting is not None:
            header[expecting] = key
            expecting = None
        else:
            print "WARNING: Unknown header key", key
    return

def pack(data, nbit):
    """downgrade data from 8bits to nbits (per value)"""
    data = data.flatten()
    if 8 % nbit != 0:
        raise ValueError("unpack: nbit must divide into 8")
    if data.dtype not in (np.uint8, np.int8):
        raise TypeError("unpack: dtype must be 8-bit")
    outdata = np.zeros(data.size / (8 / nbit)).astype('uint8')
    for index in range(1, 8 / nbit):
        outdata += data[index::8 / nbit] / (2**nbit)**index
    return outdata

def _write_data(data, nbit, file_object):
    """Writes given data to an open file, also packing if needed"""
    file_object.seek(0, 2)
    if nbit < 8:
        data = pack(data, nbit)
    data.tofile(file_object)

# TODO: Move this elsewhere?
def unpack(data, nbit):
    """upgrade data from nbits to 8bits"""
    if nbit > 8:
        raise ValueError("unpack: nbit must be <= 8")
    if 8 % nbit != 0:
        raise ValueError("unpack: nbit must divide into 8")
    if data.dtype not in (np.uint8, np.int8):
        raise TypeError("unpack: dtype must be 8-bit")
    if nbit == 8:
        return data
    elif nbit == 4:
        # Note: This technique assumes LSB-first ordering
        tmpdata = data.astype(np.int16)
        tmpdata = (tmpdata | (tmpdata <<  8)) & 0x0F0F
        tmpdata = tmpdata << 4 # Shift into high bits to avoid needing to sign extend
        updata = tmpdata
    elif nbit == 2:
        tmpdata = data.astype(np.int32)
        tmpdata = (tmpdata | (tmpdata << 16)) & 0x000F000F
        tmpdata = (tmpdata | (tmpdata <<  8)) & 0x03030303
        tmpdata = tmpdata << 6 # Shift into high bits to avoid needing to sign extend
        updata = tmpdata
    elif nbit == 1:
        tmpdata = data.astype(np.int64)
        tmpdata = (tmpdata | (tmpdata << 32)) & 0x0000000F0000000F
        tmpdata = (tmpdata | (tmpdata << 16)) & 0x0003000300030003
        tmpdata = (tmpdata | (tmpdata <<  8)) & 0x0101010101010101
        tmpdata = tmpdata << 7 # Shift into high bits to avoid needing to sign extend
        updata = tmpdata
    return updata.view(data.dtype)

class SigprocSettings(object):
    """defines, reads, writes sigproc settings"""
    def __init__(self):
        self.nifs = 0
        self.nchans = 0
        self.dtype = np.uint8
        self.nbits = 8
        self.header = {}
    def interpret_header(self):
        """redefine variables from header dictionary"""
        self.nifs = self.header['nifs']
        self.nchans = self.header['nchans']
        self.nbits = self.header['nbits']
        signed = 'signed' in self.header and self.header['signed'] is True
        if self.nbits >= 8:
            if signed:
                self.dtype = {8: np.int8,
                              16: np.int16,
                              32: np.float32,
                              64: np.float64}[self.nbits]
            else:
                self.dtype = {8: np.uint8,
                              16: np.uint16,
                              32: np.float32,
                              64: np.float64}[self.nbits]
        else:
            self.dtype = np.int8 if signed else np.uint8

class SigprocFile(SigprocSettings):
    """Reads from or writes to a sigproc filterbank file"""
    def __init__(self):
        super(SigprocFile, self).__init__()
        self.file_object = None
        self.mode = ''
        self.data = np.array([])
    def open(self, filename, mode):
        """open the filename, and read the header and data from it"""
        if 'b' not in mode:
            raise NotImplementedError("No support for non-binary files")
        self.mode = mode
        self.file_object = open(filename, mode)
        return self
    def clear(self):
        """Erases file contents"""
        self.file_object.seek(0)
        self.file_object.truncate()
    def close(self):
        """closes file object"""
        self.file_object.close()
    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
    def _find_nframe_from_file(self):
        self.read_header()
        curpos = self.file_object.tell()
        self.file_object.seek(0, 2) # Seek to end of file
        frame_bits = self.header['nifs'] * self.header['nchans'] * self.header['nbits']
        nframe = (self.file_object.tell() - curpos) * 8 // frame_bits
        return nframe
    def get_nframe(self):
        """calculate the number of frames from the data"""
        if self.data.size % self.nifs != 0:
            raise ValueError
        elif self.data.size // self.nifs % self.nchans != 0:
            raise ValueError
        nframe = self.data.size // self.nifs // self.nchans
        return nframe
    def read_header(self):
        """reads in a header from the file and sets local settings"""
        self.header = _read_header(self.file_object)
        self.interpret_header()
    def read_data(self, start=None, end=None):
        """read data from file and store it locally"""
        nframe = self._find_nframe_from_file()
        seek_to_data(self.file_object)
        read_start = 0
        end_read = nframe * self.nifs * self.nchans
        if start is not None:
            if start < 0:
                read_start = (nframe + start) * self.nifs * self.nchans
            elif start >= 0:
                read_start = start * self.nifs * self.nchans
        if end is not None:
            if end < 0:
                end_read = (nframe + end) * self.nifs * self.nchans
            elif end >= 0:
                end_read = end * self.nifs * self.nchans
        self.file_object.seek(read_start, os.SEEK_CUR)
        nbytes_to_read = end_read - read_start
        data = np.fromfile(self.file_object, count=nbytes_to_read, dtype=self.dtype)
        nframe = data.size // self.nifs // self.nchans
        data = data.reshape((nframe, self.nifs, self.nchans))
        if self.nbits < 8:
            data = unpack(data, self.nbits)
        self.data = data
        return self.data
    def write_to(self, filename):
        """writes data and header to a different file"""
        file_object = open(filename, 'wb')
        _write_header(self.header, file_object)
        _write_data(self.data, self.nbits, file_object)
    def append_data(self, input_data):
        """append data to local data and file"""
        input_frames = input_data.size // self.nifs // self.nchans
        input_shape = (input_frames, self.nifs, self.nchans)
        input_data = np.reshape(input_data.flatten(), input_shape)
        if any(character in self.mode for character in 'w+a'):
            _write_data(input_data, self.nbits, self.file_object)
        if self.data.size > 0:
            self.data = np.append(self.data.flatten(), input_data.flatten())
        else:
            self.data = input_data.flatten()
        nframe = self.get_nframe()
        frame_shape = (nframe, self.nifs, self.nchans)
        self.data = np.reshape(self.data, frame_shape)
