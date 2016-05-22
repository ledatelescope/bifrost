
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

from libbifrost import _bf, _check, _get, _string2space

def _get_space(arr):
	try:             return arr.flags['SPACE']
	except KeyError: return 'system' # TODO: Dangerous to assume?

# TODO: Any use for these two?
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
