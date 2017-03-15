
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
import bifrost.sigproc2 as sigproc

from copy import deepcopy

def get_with_default(obj, key, default=None):
	return obj[key] if key in obj else default

def mjd2unix(mjd):
	return (mjd - 40587) * 86400

class SigprocSourceBlock(SourceBlock):
	def __init__(self, filenames, gulp_nframe, unpack=True, *args, **kwargs):
		super(SigprocSourceBlock, self).__init__(filenames, gulp_nframe, *args, **kwargs)
		self.unpack = unpack
	def create_reader(self, sourcename):
		return sigproc.SigprocFile(sourcename)
	def on_sequence(self, ireader, sourcename):
		ihdr = ireader.header
		assert(ihdr['data_type'] in [1,  # filterbank
		                             2,  # (dedispersed) time series
		                             6]) # dedispersed subbands
		for coord_frame in ['pulsarcentric', 'barycentric', 'topocentric']:
			if coord_frame in ihdr and bool(ihdr[coord_frame]):
				break
		tstart_unix = mjd2unix(ihdr['tstart'])
		nbit = ihdr['nbits']
		if self.unpack:
			nbit = max(nbit, 8)
		ohdr = {
			'_tensor': {
				'dtype':  ['u','i'][ihdr['signed']] + str(nbit),
				'shape':  [-1, ihdr['nifs'], ihdr['nchans']],
				'labels': ['time', 'pol', 'freq'],
				'scales': [(tstart_unix,ihdr['tsamp']),
				           None,
				           (ihdr['fch1'],ihdr['foff'])],
				'units':  ['s', None, 'MHz']
			},
			'frame_rate': 1./ihdr['tsamp'], # TODO: Used for anything?
			'source_name':   get_with_default(ihdr, 'source_name'),
			'rawdatafile':   get_with_default(ihdr, 'rawdatafile'),
			'az_start':      get_with_default(ihdr, 'az_start'),
			'za_start':      get_with_default(ihdr, 'za_start'),
			'raj':           get_with_default(ihdr, 'src_raj'),
			'dej':           get_with_default(ihdr, 'src_dej'),
			'refdm':         get_with_default(ihdr, 'refdm', 0.),
			'refdm_units':   'pc cm^-3',
			'telescope':     get_with_default(ihdr, 'telescope_id'),
			'machine':       get_with_default(ihdr, 'machine_id'),
			'ibeam':         get_with_default(ihdr, 'ibeam'),
			'nbeams':        get_with_default(ihdr, 'nbeams'),
			'coord_frame':   coord_frame,
		}
		# Convert ids to strings
		ohdr['telescope'] = sigproc.id2telescope(ohdr['telescope'])
		ohdr['machine']   = sigproc.id2machine(ohdr['machine'])
		# Note: This gives 32 bits to the fractional part of a second,
		#         corresponding to ~0.233ns resolution. The whole part
		#         gets at least 31 bits, which will overflow in 2038.
		time_tag  = int(round(tstart_unix * 2**32))
		ohdr['time_tag'] = time_tag
		ohdr['name']     = sourcename
		return [ohdr]
	def on_data(self, reader, ospans):
		ospan = ospans[0]
		#print "SigprocReadBlock::on_data", ospan.data.dtype
		if self.unpack:
			indata = reader.read(ospan.shape[0])
			nframe = indata.shape[0]
			#print indata.shape, indata.dtype, nframe
			#print indata
			ospan.data[:nframe] = indata
			# TODO: This will break when frame size < 1 byte
			#         Can't use frame_nbyte; must use something like frame_nbit
			#           Gets tricky though because what array shape+dtype to use?
			#             Would need to use an array class that supports dtypes
			#               of any nbit, and deals with consequent indexing issues.
			#           Admittedly, it is probably a rare case, because a detected
			#             time series would typically have SNR warranting >= 8
			#             bits.
			#           Multiple pols could be included, but only if chan and pol
			#             dims are merged together.
			#print "NBYTE", nbyte
			#assert(nbyte % reader.frame_nbyte == 0)
			#nframe = nbyte // reader.frame_nbyte
		else:
			nbyte = reader.readinto(ospan.data)
			if nbyte % ospan.frame_nbyte:
				raise IOError("Input file is truncated")
			nframe = nbyte // ospan.frame_nbyte
		return [nframe]

def read_sigproc(filenames, gulp_nframe, unpack=True, *args, **kwargs):
	return SigprocSourceBlock(filenames, gulp_nframe, unpack,
	                          *args, **kwargs)
