#This program is for reading and writing sigproc filterbank files.

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

#header parameter names which precede strings
_string_values = ['source_name',
                  'rawdatafile']
#header parameter names which precede doubles
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
#header parameter names which precede integers
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
#this header parameter precedes a character
_character_values = ['signed']

#the data_type parameter names' translation
_data_types = defaultdict(lambda : 'unknown',
                          {0: 'raw data',
                           1: 'filterbank',
                           2: 'time series',
                           3: 'pulse profile',
                           4: 'amplitude spectrum',
                           5: 'complex spectrum',
                           6: 'dedispersed subbands'})
#the telescope_id parameter names' translation
_telescopes = defaultdict(lambda : 'unknown',
                          {0:  'Fake',
                           1:  'Arecibo',
                           2:  'Ooty',
                           3:  'Nancay',
                           4:  'Parkes',
                           5:  'Jodrell',#'Lovell',
                           6:  'GBT',
                           7:  'GMRT',
                           8:  'Effelsberg',
                           52: 'LWA-OV',
                           53: 'LWA-SV'})
#the machine_id parameter names' translation
_machines   = defaultdict(lambda : 'unknown',
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
		elif key == "header_size":
			pass
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


def pack(data,nbit):
	data = data.flatten()
	if 8 % nbit != 0:
		raise ValueError("unpack: nbit must divide into 8")
	if data.dtype not in (np.uint8, np.int8):
		raise TypeError("unpack: dtype must be 8-bit")
	if nbit == 4:
		data[1::2]=data[1::2]/16
		data = np.zeros(data.size/2).astype('uint8')+data[0::2]+data[1::2]
	elif nbit == 2:
		data[1::4]=data[1::4]/4
		data[2::4]=data[2::4]/16
		data[3::4]=data[3::4]/64
		data = np.zeros(data.size/4).astype('uint8')+\
				data[0::4]+data[1::4]+data[2::4]+data[3::4]
	elif nbit == 1:
		data[1::8]=data[1::8]/2
		data[2::8]=data[2::8]/4
		data[3::8]=data[3::8]/8
		data[4::8]=data[4::8]/16
		data[5::8]=data[5::8]/32
		data[6::8]=data[6::8]/64
		data[7::8]=data[7::8]/128
		data = np.zeros(data.size/8).astype('uint8')+\
				data[0::8]+data[1::8]+data[2::8]+data[3::8]+\
				data[4::8]+data[5::8]+data[6::8]+data[7::8]
	return data

def _write_data(data,nbit,f):
	if nbit<8:
		data = pack(data,nbit)
	data.tofile(f)

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
		# Note: This technique assumes LSB-first ordering
		x = data.astype(np.int16)#np.empty(upshape, dtype=np.int16)
		x = (x | (x <<  8)) & 0x0F0F
		x = x << 4 # Shift into high bits to avoid needing to sign extend
		updata = x
	elif nbit == 2:
		x = data.astype(np.int32)#np.empty(upshape, dtype=np.int16)
		x = (x | (x << 16)) & 0x000F000F
		x = (x | (x <<  8)) & 0x03030303
		x = x << 6 # Shift into high bits to avoid needing to sign extend
		updata = x
	elif nbit == 1:
		x = data.astype(np.int64)#np.empty(upshape, dtype=np.int16)
		x = (x | (x << 32)) & 0x0000000F0000000F
		x = (x | (x << 16)) & 0x0003000300030003
		x = (x | (x <<  8)) & 0x0101010101010101
		x = x << 7 # Shift into high bits to avoid needing to sign extend
		updata = x
	return updata.view(data.dtype)
	
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
		signed = 'signed' in self.header and self.header['signed'] == True
		if self.nbit >= 8:
			if signed:
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
			self.dtype = np.int8 if signed else np.uint8
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
	def seek(self, offset, whence=0):
		if whence == 0:
			offset += self.header_size
		self.f.seek(offset, whence)
	def bandwidth(self):
		return self.header['nchans'] * self.header['foff']
	def cfreq(self):
		return self.header['fch1'] + 0.5*(self.header['nchans']-1)*self.header['foff']
	def duration(self):
		return self.header['tsamp'] * self.nframe()
	def nframe(self):
		if 'nsamples' not in self.header or self.header['nsamples'] == 0:
			curpos = self.f.tell()
			self.f.seek(0, 2) # Seek to end of file
			frame_bits = self.header['nifs'] * self.header['nchans'] * self.header['nbits']
			nframe = (self.f.tell() - self.header['header_size'])*8 / frame_bits
			self.header['nsamples'] = nframe
			self.f.seek(curpos, 0) # Seek back to where we were
		return self.header['nsamples']
	def read(self, nframe_or_start, end=None):
		if end is not None:
			start = nframe_or_start or 0
			self.seek(start * self.frame_nbyte)
			if end == -1:
				end = self.nframe()
			nframe = end - start
		else:
			nframe = nframe_or_start
		data = np.fromfile(self.f, count=nframe*self.frame_size, dtype=self.dtype)
		nframe = data.size // self.frame_size
		data = data.reshape((nframe,)+self.frame_shape)
            
		nbit = self.header['nbits']
		if nbit < 8:
			data = unpack(data, nbit)
		return data
	def readinto(self, buf):
		return self.f.readinto(buf)
	def __str__(self):
		hmod = self.header.copy()
		d = hmod['data_type']
		hmod['data_type'] = "%i (%s)" % (d, _data_types[d])
		t = hmod['telescope_id']
		hmod['telescope_id'] = "%i (%s)" % (t, _telescopes[t])
		m = hmod['machine_id']
		hmod['machine_id']   = "%i (%s)" % (m, _machines[m])
		return '\n'.join(['% 16s: %s' % (key,val) for (key,val) in hmod.items()])
	def __getitem__(self, key):
		if isinstance(key, type("")): # Header key lookup
			return self.header[key]
		elif isinstance(key, int): # Extract one time slice
			return self.read(key, key+1)[0]
		elif isinstance(key, slice): # 1D slice
			start = key.start if key.start is not None else  0
			stop  = key.stop  if key.stop  is not None else -1
			data = self.read(start, stop)
			#data = self.read(stop) if start == 0 else \
			#       self.read(start, key.stop)
			return data[::key.step]
		elif isinstance(key, tuple): # ND key
			raise NotImplementedError

class SigprocFileRW(object):
	def __init__(self, filename = None, mode= ''):
		if filename is not None:
			self._filename = filename
			self._header = {}
			self._data = []
			if len(mode) > 0:
				self.open(filename,mode)
	#open the filename, and read the header and data from it
	def open(self, filename=None, mode=''):
		if filename is not None:
			self._filename = filename
		if self._filename is None:
			raise ValueError("No filename inputted.")
		if len(mode) == 0:
			raise IOError("No input/output mode set.")
		if 'b' not in mode:
			raise NotImplementedError("No support for non-binary files")
		self._header = {}
		self._data = []
		self._appending = ('a' in mode)
		self._writing = any(i in mode for i in 'w+')
		self._reading = ('r' in mode)
		self.f = open(self._filename,mode)
		self.header
		self.data
	#using our current stored header, redefine other local variables
	def _interpret_header(self):
		if 'header_size' in self._header:
			self._header_length = self._header['header_size']
		else:
			self._header_length = -1
		self.nifs = self._header['nifs']
		self.nchans = self._header['nchans']
		self.frame_shape = (self.nifs, self.nchans)
		self.nbit = self._header['nbits']
		signed = 'signed' in self._header and self._header['signed'] == True
		if self.nbit >= 8:
			if signed:
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
			self.dtype = np.int8 if signed else np.uint8
			pack_factor = 8 / self.nbit
			self.frame_shape = (self.frame_shape[0],
			                    int(np.ceil(self.frame_shape[1]/float(pack_factor))))

			#self.frame_shape[-1] /= pack_factor
		self.frame_size  = self.frame_shape[0]*self.frame_shape[1]
		self.frame_nbyte = self.frame_size*self.dtype().itemsize
		if 'nsamples' in self._header and self._header['nsamples']!=0:
			self.nframe = self._header['nsamples']
		else:
			self.nframe = self._find_nframe_from_file()
	def close(self):
		self.f.close()
	def __enter__(self):
		return self
	def __exit__(self, type, value, tb):
		self.close()
	#move along file
	def seek(self, offset, whence=0):
		if whence == 0:
			offset += self.header_size
		self.f.seek(offset, whence)
	def _find_nframe_from_file(self):
		curpos = self.f.tell()
		self.f.seek(0, 2) # Seek to end of file
		frame_bits = self.nifs*self.nchans*self.nbit
		nframe = (self.f.tell() - self.header['header_size'])*8 / frame_bits
		self.f.seek(curpos, 0) # Seek back to where we were
		return nframe		
	def _find_nframe_from_data(self):
		return self.data.shape[0]
	#get all data from file and store it locally
	def read(self, nframe=None):
		if nframe == None:
			nframe = self.nframe
		data = np.fromfile(self.f, count=nframe*self.frame_size, dtype=self.dtype)
		nframe = data.size // self.frame_size
		data = data.reshape((nframe,)+self.frame_shape)
		nbit = self.header['nbits']
		if nbit < 8:
			data = unpack(data, nbit)
		self._data = data
	#appends to local data, not the file
	def append_data(self,input_data):
		self._data = np.append(self._data,input_data)
		self.nframe = self._find_nframe_from_data()
	def write_header_to(self,f):
		_write_header(self.header,f)
	def write_data_to(self,f):
		_write_data(self._data,self.nbit,f)
	#writes stored header and data to file
	def write_to(self,filename):
		f = open(filename,'wb')
		self.write_header_to(f)
		self.write_data_to(f)
	#check if should read data from file before returning
	@property
	def data(self):
		if len(self._header)==0 and self._reading:
			self.header
		if len(self._header)!=0 and\
			len(self._data) == 0 and\
			self._reading:
				self.read()
		return self._data
	@data.setter
	def data(self, input_data):
		self._data = input_data
	#reads header from file if not already set.
	@property
	def header(self):
		if len(self._header)==0 and self._reading:
			self._header = _read_header(self.f)
			self._interpret_header()
		return self._header
	#reinterprets header after setting
	@header.setter
	def header(self, input_header):
		self._header = input_header
		self._interpret_header()

