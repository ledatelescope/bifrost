
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

from libbifrost import _bf, _check, _get, _array
import bifrost as bf
import numpy as np
import ctypes

def map(func_string, shape=None,# axis_names=None,
        *args,
        **kwargs):
	"""Apply a function to a set of ndarrays.
	
	Arguments:
	  func_string: The function to apply to the arrays, as a string (see below
	               for examples).
	  shape:       The shape of the computation.
	  *args:       List of string names by which each axis is referenced
	               in func_string.
	  **kwargs:    Map of string names to ndarrays.
	
	If shape is None, the broadcast shape of all of the arrays is used.
	
	Examples:
	  # Add two arrays together
	  bf.map("c = a + b", c=c, a=a, b=b)
	
	  # Compute outer product of two arrays
	  bf.map("c(i,j) = a(i) * b(j)", 'i', 'j', c=c, a=a, b=b)
	
	  # Split the components of a complex array
	  bf.map("a = c.real; b = c.imag", c=c, a=a, b=b)
	
	  # Raise an array to a scalar power
	  bf.map("c = pow(a, p)", c=c, a=a, p=2.0)
	
	  # Slice an array with a scalar index
	  bf.map("c(i) = a(i,k)", 'i', c=c, a=a, k=7, shape=c.shape)
	"""
	#if 'shape' in kwargs:
	#	shape = kwargs.pop('shape')
	#else:
	#	shape = None
	if isinstance(shape, basestring):
		raise TypeError("Invalid type for shape argument")
	if any([not isinstance(arg, basestring) for arg in args]):
		raise TypeError("Invalid type for index name, must be string")
	axis_names = args
	#if axis_names is None:
	#	# TODO: If this is desirable, move it into the backend instead
	#	axis_names = ['_%i'%i for i in xrange(len(shape))]
	#else:
	#if len(axis_names) != len(shape):
	#	raise ValueError('Number of axis names must match number of dims in shape')
	ndim      = len(shape) if shape is not None else 0
	narg      = len(kwargs)
	def is_literal(x):
		return (isinstance(x, int) or
		        isinstance(x, float) or
		        isinstance(x, complex))
	arg_arrays = []
	args = []
	arg_names = []
	for key,arg in kwargs.items():
		if is_literal(arg):
			arr = np.array(arg)
			if isinstance(arg, int) and -(1<<31) <= arg < (1<<31):
				arr = arr.astype(np.int32)
			# TODO: Any way to decide when these should be double-precision?
			elif isinstance(arg, float):
				arr = arr.astype(np.float32)
			elif isinstance(arg, complex):
				arr = arr.astype(np.complex64)
			arr.flags['WRITEABLE'] = False
			arr = bf.asarray(arr)
		else:
			arr = bf.asarray(arg)
		# Note: We must keep a reference to each array lest they be garbage
		#         collected before their corresponding BFarray is used.
		arg_arrays.append(arr)
		args.append(arr.as_BFarray())
		arg_names.append(key)
	_check(_bf.Map(ndim, _array(shape, dtype=ctypes.c_long), _array(axis_names),
	               narg, _array(args), _array(arg_names),
	               func_string))
