
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

from bifrost.pipeline import block_view
from ..DataType import DataType

from copy import deepcopy

def split_axis(block, axis, n, label=None):
	# Set function attributes to enable capture in nested function (closure)
	split_axis.axis  = axis
	split_axis.n     = n
	split_axis.label = label
	def header_transform(hdr):
		axis  = split_axis.axis
		n     = split_axis.n
		label = split_axis.label
		hdr = deepcopy(hdr)
		tensor = hdr['_tensor']
		if isinstance(axis, basestring):
			axis = tensor['labels'].index(axis)
		shape = tensor['shape']
		if shape[axis] == -1:
			# Axis is frame axis
			# TODO: Should assert even division here instead?
			hdr['gulp_nframe'] = (hdr['gulp_nframe']-1)/n+1
		else:
			# Axis is not frame axis
			if shape[axis] % n:
				raise ValueError("Split does not evenly divide axis (%i // %i)" %
				                 tensor['shape'][axis], n)
			shape[axis] //= n
		shape.insert(axis+1, n)
		tensor['units'].insert(axis+1, tensor['units'][axis])
		tensor['labels'].insert(axis+1, label)
		tensor['scales'].insert(axis+1, [0,tensor['scales'][axis][1]])
		tensor['scales'][axis][1] /= n
		return hdr
	return block_view(block, header_transform)

def astype(block, dtype):
	astype.dtype = dtype
	def header_transform(hdr):
		new_dtype = astype.dtype
		hdr = deepcopy(hdr)
		tensor = hdr['_tensor']
		old_dtype = tensor['dtype']
		old_itemsize = DataType(old_dtype).itemsize
		new_itemsize = DataType(new_dtype).itemsize
		old_axissize = old_itemsize * tensor['shape'][-1]
		if old_axissize % new_itemsize:
			raise ValueError("New type not compatible with data shape")
		tensor['shape'][-1] = old_axissize // new_itemsize
		tensor['dtype'] = dtype
		return hdr
	return block_view(block, header_transform)
