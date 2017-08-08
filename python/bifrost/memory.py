
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
import ctypes

def space_accessible(space, from_spaces):
    if from_spaces == 'any': # TODO: This is a little bit hacky
        return True
    from_spaces = set(from_spaces)
    if space in from_spaces:
        return True
    elif space == 'cuda_host':
        return 'system' in from_spaces
    elif space == 'cuda_managed':
        return 'system' in from_spaces or 'cuda' in from_spaces
    else:
        return False

def raw_malloc(size, space):
    ptr = ctypes.c_void_p()
    _check(_bf.bfMalloc(ptr, size, _string2space(space)))
    return ptr.value
def raw_free(ptr, space='auto'):
    _check(_bf.bfFree(ptr, _string2space(space)))
def raw_get_space(ptr):
    return _get(_bf.bfGetSpace, ptr)

def alignment():
    ret, _ = _bf.bfGetAlignment()
    return ret

# **TODO: Deprecate below here!

def _get_space(arr):
    try:             return arr.flags['SPACE']
    except KeyError: return 'system' # TODO: Dangerous to assume?

# Note: These functions operate on numpy or GPU arrays
def memcpy(dst, src):
    assert(dst.flags['C_CONTIGUOUS'])
    assert(src.shape == dst.shape)
    dst_space = _string2space(_get_space(dst))
    src_space = _string2space(_get_space(src))
    count = dst.nbytes
    _check(_bf.bfMemcpy(dst.ctypes.data, dst_space,
                        src.ctypes.data, src_space,
                        count))
    return dst
def memcpy2D(dst, src):
    assert(len(dst.shape) == 2)
    assert(src.shape == dst.shape)
    dst_space = _string2space(_get_space(dst))
    src_space = _string2space(_get_space(src))
    height, width = dst.shape
    width_bytes = width * dst.dtype.itemsize
    _check(_bf.bfMemcpy2D(dst.ctypes.data, dst.strides[0], dst_space,
                          src.ctypes.data, src.strides[0], src_space,
                          width_bytes, height))
def memset(dst, val=0):
    assert(dst.flags['C_CONTIGUOUS'])
    space = _string2space(_get_space(dst))
    count = dst.nbytes
    _check(_bf.bfMemset(dst.ctypes.data, space, val, count))
def memset2D(dst, val=0):
    assert(len(dst.shape) == 2)
    space = _string2space(_get_space(dst))
    height, width = dst.shape
    width_bytes = width * dst.dtype.itemsize
    _check(_bf.bfMemset2D(dst.ctypes.data, dst.strides[0], space,
                          val, width_bytes, height))
