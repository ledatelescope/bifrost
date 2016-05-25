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


#  Copyright 2015 Ben Barsdell
#  Copyright 2016 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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

_telescopes = defaultdict(lambda : "unknown",
                          {0:  'fake',
                           1:  'Arecibo',
                           2:  'Ooty',
                           4:  'Parkes',
                           5:  'Lovell',
                           7:  'GMRT',
                           52: 'LWA-OV'})
_machines   = defaultdict(lambda : "unknown",
                          {0:  'FAKE',
                           1:  'PSPM',
                           2:  'WAPP',
                           3:  'OOTY',
                           52: 'LWA-OV'})

def _header_write_string(f, key):
	f.write(struct.pack('=i', len(key)))
	f.write(key)
def _header_write(f, key, value):
	if isinstance(value, int):
		fmt = '=i'
	elif isinstance(value, float):
		fmt = '=d'
	elif key == 'signed':
		fmt = '=b'
	else:
		raise TypeError("Invalid value type")
	_header_write_string(f, key)
	f.write(struct.pack(fmt, value))

def _header_read(f):
	length = struct.unpack('=i', f.read(4))[0]
	if length <= 0 or length >= 80:
		return None
	s = f.read(length)
	return s

def _write_header(hdr, f):
	#f.write("HEADER_START")
	_header_write_string(f, "HEADER_START")
	for key,val in hdr.items():
		if key in _string_values:
			_header_write_string(f, key)
			_header_write_string(f, val)
		elif key in _double_values:
			_header_write(f, key, float(val))
		elif key in _integer_values:
			_header_write(f, key, int(val))
		#elif key in _character_values:
		#	_header_write(f, key, ??
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
	#	f.seek(0, 2) # Seek to end of file
	#	header['nsamples'] = (f.tell() - header['header_size'])*8 / frame_bits
	#	f.seek(header['header_size'], 0) # Seek back to end of header
	return header

# TODO: Add support for writing
class SigprocFile(object):
	def __init__(self, filename=None):
		if filename is not None:
			self.open(filename)
	def open(self, filename):
		# Note: If nbit < 8, pack_factor = 8 / nbit and the last dimension
		#         is divided by pack_factor, with dtype set to uint8.
		self.f = open(filename, 'rb')
		self.header = _read_header(self.f)
		self.frame_shape = (self.header['nifs'], self.header['nchans'])
		self.nbit = self.header['nbits']
		if self.nbit >= 8:
			if 'signed' in self.header and self.header['signed'] == True:
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
			self.dtype = np.uint8
			pack_factor = 8 / self.nbit
			self.frame_shape = (self.frame_shape[0],
			                    self.frame_shape[1]/pack_factor)
			#self.frame_shape[-1] /= pack_factor
		self.frame_size  = self.frame_shape[0]*self.frame_shape[1]
		self.frame_nbyte = self.frame_size*self.dtype().itemsize
		return self
	def close(self):
		self.f.close()
	def __enter__(self):
		return self
	def __exit__(self, type, value, tb):
		self.close()
	def read(self, nframe):
		data = np.fromfile(self.f, count=nframe*self.frame_size, dtype=self.dtype)
		data = data.reshape((nframe,)+self.frame_shape)
		return data
	def readinto(self, buf):
		return self.f.readinto(buf)
	def __str__(self):
		hmod = h.copy()
		t = hmod['telescope_id']
		hmod['telescope_id'] = "%i (%s)" % (t, _telescopes[t])
		m = hmod['machine_id']
		hmod['machine_id']   = "%i (%s)" % (m, _machines[m])
		return '\n'.join(['% 16s: %s' % (key,val) for (key,val) in hmod.items()])
