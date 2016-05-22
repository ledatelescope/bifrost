
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

from libbifrost import _bf, _check, _get, _string2space

def _get_space(arr):
	try:             return arr.flags['SPACE']
	except KeyError: return 'system' # TODO: Dangerous to assume?

def raw_malloc(size, space):
	return _get(_bf.Malloc(size=size, space=_string2space(space)), retarg=0)
def raw_free(ptr, space='auto'):
	_check(_bf.Free(ptr, _string2space(space)))
def raw_get_space(ptr):
	# TODO: bfGetSpace currently has a funny call signature
	raise NotImplemented("bfGetSpace")

def alignment():
	ret, _ = _bf.GetAlignment()
	return ret

# Note: These functions operate on numpy or GPU arrays
def memcpy(dst, src):
	assert(dst.flags['C_CONTIGUOUS'])
	assert(src.shape == dst.shape)
	dst_space = _string2space(_get_space(dst))
	src_space = _string2space(_get_space(src))
	count = dst.nbytes
	_check(_bf.Memcpy(dst.ctypes.data, dst_space,
	                  src.ctypes.data, src_space,
	                  count))
	return dst
def memcpy2D(dst, src):
	assert(len(dst.shape) == 2)
	assert(src.shape == dst.shape)
	dst_space = _string2space(_get_space(dst))
	src_space = _string2space(_get_space(src))
	_check(_bf.Memcpy2D(dst.ctypes.data, dst.strides[0], dst_space,
	                    src.ctypes.data, src.strides[0], src_space,
	                    dst.shape[1], dst.shape[0]))
def memset(dst, val=0):
	assert(dst.flags['C_CONTIGUOUS'])
	space = _string2space(_get_space(dst))
	count = dst.nbytes
	_check(_bf.Memset(dst.ctypes.data, space, val, count))
def memset2D(dst, val=0):
	assert(len(dst.shape) == 2)
	space = _string2space(_get_space(dst))
	_check(_bf.Memset2D(dst.ctypes.data, dst.strides[0], space,
	                    val, dst.shape[1], dst.shape[0]))
def transpose(dst, src, axes=None):
	dst_space = _string2space(_get_space(dst))
	src_space = _string2space(_get_space(src))
	assert( dst_space    == src_space )
	assert( dst.itemsize == src.itemsize )
	assert( dst.ndim     == src.ndim )
	if axes is None:
		# Default to reversing the dims
		axes = [src.ndim-1-d for d in xrange(src.ndim)]
	assert( len(axes) == src.ndim )
	arraytype = _bf.BFsize*src.ndim
	dst_strides = arraytype(*dst.strides)
	src_strides = arraytype(*src.strides)
	shape_array = arraytype(*src.shape)
	axes_array  = arraytype(*axes)
	_check(_bf.Transpose(dst.ctypes.data, dst_strides,
	                     src.ctypes.data, src_strides,
	                     src_space,
	                     src.itemsize,
	                     src.ndim,
	                     shape_array,
	                     axes_array))
