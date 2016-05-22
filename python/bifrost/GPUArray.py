
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

from memory import raw_malloc, raw_free, memset, memcpy, memcpy2D

import numpy as np

class GPUArray(object):
	def __init__(self, shape, dtype, buffer=None, offset=0, strides=None):
		itemsize = dtype().itemsize
		if strides is None:
			# This magic came from http://stackoverflow.com/a/32874295
			strides = itemsize*np.r_[1,np.cumprod(shape[::-1][:-1])][::-1]
		self.shape   = shape
		self.dtype   = dtype
		self.buffer  = buffer
		self.offset  = offset
		self.strides = strides
		self.base    = None
		self.flags   = {'WRITEABLE':    True,
		                'ALIGNED':      buffer%itemsize==0 if buffer is not None else True,
		                'OWNDATA':      False,
		                'UPDATEIFCOPY': False,
		                'C_CONTIGUOUS': self.nbytes==strides[0]*shape[0],
		                'F_CONTIGUOUS': False,
		                'SPACE':        'cuda'}
		class CTypes(object):
			def __init__(self, parent):
				self.parent = parent
			@property
			def data(self):
				return self.parent.data
		self.ctypes = CTypes(self)
		if self.buffer is None:
			self.buffer = raw_malloc(self.nbytes, space='cuda')
			self.flags['OWNDATA'] = True
			self.flags['ALIGNED'] = True
			memset(self, 0)
		else:
			self.buffer += offset
	def __del__(self):
		if self.flags['OWNDATA']:
			raw_free(self.buffer, self.flags['SPACE'])
	@property
	def data(self):
		return self.buffer
	#def reshape(self, shape):
	#	# TODO: How to deal with strides?
	#	#         May be non-contiguous but the reshape still works
	#	#           E.g., splitting dims
	#	return GPUArray(shape, self.dtype,
	#	                buffer=self.buffer,
	#	                offset=self.offset,
	#	                strides=self.strides)
	@property
	def size(self):
		return int(np.prod(self.shape))
	@property
	def itemsize(self):
		return self.dtype().itemsize
	@property
	def nbytes(self):
		return self.size*self.itemsize
	@property
	def ndim(self):
		return len(self.shape)
	def get(self, dst=None):
		#hdata = dst if dst is not None else np.empty(self.shape, self.dtype)
		hdata = dst if dst is not None else np.zeros(self.shape, self.dtype)
		if self.flags['C_CONTIGUOUS']:
			memcpy(hdata, self)
		elif self.ndim == 2:
			memcpy2D(hdata, self)
		else:
			raise RuntimeError("Copying with this data layout is unsupported")
		return hdata
	def set(self, hdata):
		assert(hdata.shape == self.shape)
		if self.flags['C_CONTIGUOUS'] and hdata.flags['C_CONTIGUOUS']:
			memcpy(self, hdata)
		elif self.ndim == 2:
			memcpy2D(self, hdata)
		else:
			raise RuntimeError("Copying with this data layout is unsupported")
		return self
