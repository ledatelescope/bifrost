
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

from bifrost.pipeline import TransformBlock
from bifrost.fft import Fft
from bifrost.units import transform_units
from bifrost.DataType import DataType

from copy import deepcopy
import math

class FftBlock(TransformBlock):
	"""This block produces an N-dimensional FFT of the input data stream. The
	transform can be over any set of dimensions that does NOT include the frame
	(time) dimension. Transforms over the frame dimension can be achieved by
	first reshaping the input data stream.
	Axis scales are automatically updated to reflect the Fourier-transformed
	axes.
	"""
	# TODO: Add support for sizes (aka 's') parameter that defines transform
	#         length in each dimension (i.e., cropped/padded transforms).
	#         Should be able to do this using an input callback and padded
	#           output dims.
	def __init__(self, iring, axes, inverse=False, real_output=False,
	             axis_labels=None,
	             *args, **kwargs):
		super(FftBlock, self).__init__(iring, *args, **kwargs)
		if not isinstance(axes, list) or isinstance(axes, tuple):
			axes = [axes]
		if not isinstance(axis_labels, list) or isinstance(axis_labels, tuple):
			axis_labels = [axis_labels]
		self.specified_axes = axes
		self.real_output = real_output
		self.inverse     = inverse
		self.axis_labels = axis_labels
		self.space       = self.irings[0].space
		self.fft         = Fft()
	def define_valid_input_spaces(self):
		"""Return set of valid spaces (or 'any') for each input"""
		return ('cuda',)
	def on_sequence(self, iseq):
		ihdr = iseq.header
		itensor = ihdr['_tensor']
		# TODO: DataType cast should be done inside ring2
		#         **This tensor stuff generally needs to be cleaned up
		itype = DataType(itensor['dtype'])
		# TODO: This is slightly hacky; it needs to emulate the type casting
		#         that Bifrost does internally for the FFT.
		itype = itype.as_floating_point()
		
		# Get axis indices, allowing for lookup-by-label
		self.axes = [itensor['labels'].index(axis)
		             if isinstance(axis, basestring)
		             else axis
		             for axis in self.specified_axes]
		
		axes = self.axes
		shape = [itensor['shape'][ax] for ax in axes]
		
		otype = itype.as_real() if self.real_output else itype.as_complex()
		ohdr = deepcopy(ihdr)
		otensor = ohdr['_tensor']
		otensor['dtype'] = str(otype)
		if itype.is_real and otype.is_complex:
			self.mode = 'r2c'
		elif itype.is_complex and otype.is_real:
			self.mode = 'c2r'
		else:
			self.mode = 'c2c'
		frame_axis = itensor['shape'].index(-1)
		if frame_axis in axes:
			raise KeyError("Cannot transform frame axis; reshape the data stream first")
		
		# Adjust output shape for real transforms
		if self.mode == 'r2c':
			otensor['shape'][axes[-1]] //= 2
			otensor['shape'][axes[-1]]  += 1
		elif self.mode == 'c2r':
			otensor['shape'][axes[-1]]  -= 1
			otensor['shape'][axes[-1]]  *= 2
			shape[-1] -= 1
			shape[-1] *= 2
		
		for i, (ax, length) in enumerate(zip(axes, shape)):
			if 'units' in otensor:
				units = otensor['units'][ax]
				otensor['units'][ax] = transform_units(units, -1)
			if 'scales' in otensor:
				otensor['scales'][ax][0] = 0 # TODO: Is this OK?
				scale = otensor['scales'][ax][1]
				otensor['scales'][ax][1] = 1. / (scale*length)
			if 'labels' in otensor and self.axis_labels is not None:
				otensor['labels'][ax] = self.axis_labels[i]
		self.nframe  = 0
		self.istride = 0
		self.ostride = 0
		return ohdr
	def on_data(self, ispan, ospan):
		# Check if nframe or ring strides have changed
		if (ispan.nframe != self.nframe or
		    ispan._stride_bytes != self.istride or
		    ospan._stride_bytes != self.ostride):
			# (Re-)generate the FFT plan
			self.fft.init(ispan.data, ospan.data, axes=self.axes)
			self.nframe  = ispan.nframe
			self.istride = ispan._stride_bytes
			self.ostride = ospan._stride_bytes
		size = self.fft.workspace_size
		with self.get_temp_storage(self.space).allocate(size) as workspace:
			self.fft.execute_workspace(ispan.data, ospan.data,
			                           workspace.ptr, workspace.size,
			                           inverse=self.inverse)

def fft(iring, axes, *args, **kwargs):
	return FftBlock(iring, axes, *args, **kwargs)
