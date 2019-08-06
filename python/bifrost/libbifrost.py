
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

# This file provides a direct interface to libbifrost.so

# PYCLIBRARY ISSUE: Passing the wrong handle type to a function gives this
#                     meaningless error:
#  ArgumentError: argument 1: <type 'exceptions.TypeError'>: expected LP_s
#    instance instead of LP_s
#  E.g., _bf.bfRingSequenceGetName(<BFspan>) [should be <BFsequence>]

import ctypes
import libbifrost_generated as _bf
bf = _bf # Public access to library

# Internal helpers below

class BifrostObject(object):
    """Base class for simple objects with create/destroy functions"""
    def __init__(self, constructor, destructor, *args):
        self.obj = destructor.argtypes[0]()
        _check(constructor(ctypes.byref(self.obj), *args))
        self._destructor = destructor
    def _destroy(self):
        if self.obj:
            _check(self._destructor(self.obj))
            self.obj.values = 0
    def __del__(self):
        self._destroy()
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self._destroy()

def _array(size_or_vals, dtype=None):
    import ctypes
    if size_or_vals is None:
        return None
    try:
        _ = iter(size_or_vals)
    except TypeError:
        # Not iterable, so assume it's the size and create an empty array
        size = size_or_vals
        return (dtype * size)()
    else:
        # Iterable, so convert it to a ctypes array
        vals = size_or_vals
        if len(vals) == 0:
            return None
        if dtype is None:
            # Try to deduce type
            if isinstance(vals[0], int):
                dtype = ctypes.c_int
            elif isinstance(vals[0], float):
                dtype = ctypes.c_double
            elif isinstance(vals[0], basestring):
                dtype = ctypes.c_char_p
            elif isinstance(vals[0], _bf.BFarray):
                dtype = ctypes.POINTER(_bf.BFarray)
                vals = [ctypes.pointer(val) for val in vals]
            # else:
            #    dtype = type(vals[0])
            else:
                raise TypeError("Cannot deduce C type from ", type(vals[0]))
        return (dtype * len(vals))(*vals)

def _check(status):
    if __debug__:
        if status != _bf.BF_STATUS_SUCCESS:
            if status is None:
                raise RuntimeError("WTF, status is None")
            if status == _bf.BF_STATUS_END_OF_DATA:
                raise StopIteration()
            elif status == _bf.BF_STATUS_WOULD_BLOCK:
                raise IOError('BF_STATUS_WOULD_BLOCK')
            else:
                status_str = _bf.bfGetStatusString(status)
                raise RuntimeError(status_str)
    else:
        if status == _bf.BF_STATUS_END_OF_DATA:
            raise StopIteration()
        elif status == _bf.BF_STATUS_WOULD_BLOCK:
            raise IOError('BF_STATUS_WOULD_BLOCK')
    return status

DEREF = {ctypes.POINTER(t): t for t in [ctypes.c_bool,
                                        ctypes.c_char,
                                        ctypes.c_char_p,
                                        ctypes.c_float,
                                        ctypes.c_double,
                                        ctypes.c_longdouble,
                                        ctypes.c_int,
                                        ctypes.c_int8,
                                        ctypes.c_int16,
                                        ctypes.c_int32,
                                        ctypes.c_int64,
                                        ctypes.c_long,
                                        ctypes.c_longlong,
                                        ctypes.c_short,
                                        ctypes.c_size_t,
                                        ctypes.c_ssize_t,
                                        ctypes.c_uint,
                                        ctypes.c_uint8,
                                        ctypes.c_uint16,
                                        ctypes.c_uint32,
                                        ctypes.c_uint64,
                                        ctypes.c_ulong,
                                        ctypes.c_ulonglong,
                                        ctypes.c_ushort,
                                        ctypes.c_void_p,
                                        ctypes.c_wchar,
                                        ctypes.c_wchar_p]}
def _get(func, *args):
    retarg = -1
    dtype = DEREF[func.argtypes[retarg]]
    ret = dtype()
    args += (ctypes.byref(ret),)
    _check(func(*args))
    return ret.value

STRING2SPACE = {'auto':         _bf.BF_SPACE_AUTO,
                'system':       _bf.BF_SPACE_SYSTEM,
                'cuda':         _bf.BF_SPACE_CUDA,
                'cuda_host':    _bf.BF_SPACE_CUDA_HOST,
                'cuda_managed': _bf.BF_SPACE_CUDA_MANAGED}
def _string2space(s):
    if s not in STRING2SPACE:
        raise KeyError("Invalid space '" + str(s) +
                       "'.\nValid spaces: " + str(LUT.keys()))
    return STRING2SPACE[s]

SPACE2STRING = {_bf.BF_SPACE_AUTO:         'auto',
                _bf.BF_SPACE_SYSTEM:       'system',
                _bf.BF_SPACE_CUDA:         'cuda',
                _bf.BF_SPACE_CUDA_HOST:    'cuda_host',
                _bf.BF_SPACE_CUDA_MANAGED: 'cuda_managed'}
def _space2string(i):
    return SPACE2STRING[i]
