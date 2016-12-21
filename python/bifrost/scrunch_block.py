
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

# TODO: This is a bit hacky and inflexible, and has no CUDA backend yet

from pipeline import TransformBlock

from copy import deepcopy

class ScrunchBlock(TransformBlock):
	def __init__(self, iring, factor, *args, **kwargs):
		super(ScrunchBlock, self).__init__([iring], *args, **kwargs)
		assert(type(factor) == int)
		self.factor = factor
	def define_valid_input_spaces(self):
		"""Return set of valid spaces (or 'any') for each input"""
		return [('system',)]
	def define_output_nframes(self, input_nframes):
		"""Return output nframe for each output, given input_nframes.
		"""
		if any([nframe % self.factor != 0 for nframe in input_nframes]):
			raise ValueError("Scrunch factor does not divide gulp size")
		return [nframe//self.factor for nframe in input_nframes]
	def on_sequence(self, iseqs):
		ohdr = deepcopy(iseqs[0].header)
		ohdr['_tensor']['scales'][0][1] *= self.factor
		return [ohdr], [None]
	def on_data(self, ispans, ospans):
		in_nframe = ispans[0].nframe
		out_nframe = in_nframe // self.factor
		idata = ispans[0].data
		ospans[0].data[...] = idata.reshape((out_nframe,self.factor)+idata.shape[1:]).mean(axis=1)
		return [out_nframe]

def scrunch(iring, factor, *args, **kwargs):
	return ScrunchBlock(iring, factor, *args, **kwargs).orings[0]
