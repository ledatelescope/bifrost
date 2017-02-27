
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

from __future__ import absolute_import

from bifrost.pipeline import SourceBlock
import bifrost.guppi_raw as guppi_raw

import numpy as np

def get_with_default(obj, key, default=None):
	return obj[key] if key in obj else default

def mjd2unix(mjd):
	return (mjd - 40587) * 86400

class GuppiRawSourceBlock(SourceBlock):
	def __init__(self, sourcenames, gulp_nframe=1, *args, **kwargs):
		super(GuppiRawSourceBlock, self).__init__(sourcenames,
		                                          gulp_nframe=gulp_nframe,
		                                          *args, **kwargs)
	def create_reader(self, sourcename):
		return open(sourcename, 'rb')
	def on_sequence(self, reader, sourcename):
		previous_pos = reader.tell()
		ihdr = guppi_raw.read_header(reader)
		header_size = reader.tell() - previous_pos
		self.header_buf = bytearray(header_size)
		nbit      = ihdr['NBITS']
		assert(nbit in set([4,8,16,32,64]))
		nchan     = ihdr['OBSNCHAN']
		bw_MHz    = ihdr['OBSBW']
		cfreq_MHz = ihdr['OBSFREQ']
		df_MHz = bw_MHz / nchan
		f0_MHz = cfreq_MHz - 0.5*(nchan-1)*df_MHz
		dt_s   = 1. / df_MHz / 1e6
		# Derive the timestamp of this block
		byte_offset   = ihdr['PKTIDX'] * ihdr['PKTSIZE']
		frame_nbyte   = ihdr['BLOCSIZE'] / ihdr['NTIME']
		bytes_per_sec = frame_nbyte / dt_s
		offset_secs   = byte_offset / bytes_per_sec
		tstart_mjd    = ihdr['STT_IMJD'] + (ihdr['STT_SMJD'] + offset_secs) / 86400.
		tstart_unix   = mjd2unix(tstart_mjd)
		ohdr = {
			'_tensor': {
				'dtype':  'ci' + str(nbit),
				'shape':  [-1, nchan, ihdr['NTIME'], ihdr['NPOL']],
				# Note: 'block' is the frame axis
				'labels': ['block', 'frequency', 'time', 'polarisation'],
				'scales': [(tstart_unix, dt_s*ihdr['NTIME']),
				           (f0_MHz, df_MHz),
				           (0, dt_s),
				           None],
				'units':  ['s', 'MHz', 's', None]
			},
			'az_start':      get_with_default(ihdr, 'AZ'),            # Decimal degrees
			'za_start':      get_with_default(ihdr, 'ZA'),            # Decimal degrees
			'raj':           get_with_default(ihdr, 'RA')*(24./360.), # Decimal hours
			'dej':           get_with_default(ihdr, 'DEC'),           # Decimal degrees
			'source_name':   get_with_default(ihdr, 'SRC_NAME'),
			'refdm':         get_with_default(ihdr, 'CHAN_DM'),
			'telescope_id':  get_with_default(ihdr, 'TELESCOP'),
			'machine_id':    get_with_default(ihdr, 'BACKEND'),
			'rawdatafile':   sourcename,
			'coord_frame':   'topocentric',
		}
		# Note: This gives 32 bits to the fractional part of a second,
		#         corresponding to ~0.233ns resolution. The whole part
		#         gets at least 31 bits, which will overflow in 2038.
		time_tag  = int(round(tstart_unix * 2**32))
		ohdr['time_tag'] = time_tag
		self.already_read_header = True
		
		ohdr['name'] = sourcename
		return [ohdr]
	def on_data(self, reader, ospans):
		if not self.already_read_header:
			# Skip over header
			#ihdr = guppi_raw.read_header(reader)
			nbyte = reader.readinto(self.header_buf)
			if nbyte == 0:
				return [0] # EOF
			elif nbyte < len(self.header_buf):
				raise IOError("Block header is truncated")
		self.already_read_header = False
		ospan = ospans[0]
		odata = ospan.data
		nbyte = reader.readinto(odata)
		if nbyte % ospan.frame_nbyte:
			raise IOError("Block data is truncated")
		nframe = nbyte // ospan.frame_nbyte
		return [nframe]

def read_guppi_raw(filenames, *args, **kwargs):
	return GuppiRawSourceBlock(filenames, *args, **kwargs)
