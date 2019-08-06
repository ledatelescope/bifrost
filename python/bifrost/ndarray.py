
# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
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

"""
A np.ndarray subclass that adds support for different spaces and
  bifrost-specific metadata.

TODO: Need backend support for broadcasting arrays
TODO: Convert dtype of val in __setitem__
TODO: Some calls result in segfault with space=cuda (e.g., __getitem__
        returning scalar).

"""

import ctypes
import numpy as np
from memory import raw_malloc, raw_free, raw_get_space, space_accessible
from bifrost.libbifrost import _bf, _check
import device
from DataType import DataType
from Space import Space
import sys

# TODO: The stuff here makes array.py redundant (and outdated)

# TODO: ndarray.flags['WRITEABLE'] does not get preserved
#         through ndarray.__init__()

# TODO: Single-element assignment appears to be broken

def _address_as_buffer(address, nbyte, readonly=False):
    if address is None:
        raise ValueError("Cannot create buffer from NULL pointer")
    # Note: This doesn't work as a buffer when using pypy
    # return (ctypes.c_byte*nbyte).from_address(address)
    # Note: This works as a buffer in regular python and pypy
    # Note: int_asbuffer is undocumented; see here:
    # https://mail.scipy.org/pipermail/numpy-discussion/2008-January/030938.html
    return np.core.multiarray.int_asbuffer(
        address, nbyte, readonly=readonly, check=False)

def asarray(arr, space=None):
    if isinstance(arr, ndarray) and (space is None or space == arr.bf.space):
        return arr
    else:
        return ndarray(arr, space=space)

def empty_like(arr, space=None):
    arr = asarray(arr)
    if space is None:
        space = arr.bf.space
    return ndarray(shape=arr.shape, dtype=arr.bf.dtype, space=space,
                   native=arr.bf.native, conjugated=arr.bf.conjugated)
def empty(shape, dtype='f32', space=None, **kwargs):
    return ndarray(shape=shape, dtype=dtype, space=space, **kwargs)

def zeros_like(arr, space=None):
    ret = empty_like(arr, space)
    memset_array(ret, 0)
    return ret
def zeros(shape, dtype='f32', space=None, **kwargs):
    ret = empty(shape, dtype, space, **kwargs)
    memset_array(ret, 0)
    return ret

def copy_array(dst, src):
    dst_bf = asarray(dst)
    src_bf = asarray(src)
    if (space_accessible(dst_bf.bf.space, ['system']) and
        space_accessible(src_bf.bf.space, ['system'])):
        np.copyto(dst_bf, src_bf)
    else:
        _check(_bf.bfArrayCopy(dst_bf.as_BFarray(),
                               src_bf.as_BFarray()))
        if dst_bf.bf.space != src_bf.bf.space:
            # TODO: Decide where/when these need to be called
            device.stream_synchronize()
    return dst

def memset_array(dst, value):
    dst_bf = asarray(dst)
    _check(_bf.bfArrayMemset(dst_bf.as_BFarray(), value))
    return dst

# Stores Bifrost-specific metadata that augments Numpy's metadata
class BFArrayInfo(object):
    def __init__(self, space, dtype, native, conjugated, ownbuffer=None):
        self.space      = space
        self.dtype      = dtype
        self.native     = native
        self.conjugated = conjugated
        self.ownbuffer  = ownbuffer

# A np.ndarray subclass that adds support for different spaces and
#   bifrost-specific metadata.
# See https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
class ndarray(np.ndarray):
    def __new__(cls, base=None, space=None, shape=None, dtype=None,
                buffer=None, offset=0, strides=None,
                native=None, conjugated=None):
        if isinstance(shape, int):
            shape = [shape]
        ownbuffer = None
        if base is not None:
            if (shape is not None or
                # dtype is not None or
                buffer is not None or
                offset != 0 or
                strides is not None or
                native is not None):
                raise ValueError('Invalid combination of arguments when base '
                                 'is specified')
            if 'pycuda' in sys.modules:
                from pycuda.gpuarray import GPUArray as pycuda_GPUArray
                if isinstance(base, pycuda_GPUArray):
                    return ndarray.__new__(cls,
                                           space='cuda',
                                           buffer=int(base.gpudata),
                                           shape=base.shape,
                                           dtype=base.dtype,
                                           strides=base.strides,
                                           native=np.dtype(base.dtype).isnative)
            if dtype is not None:
                dtype = DataType(dtype)
            if space is None and dtype is None:
                if not isinstance(base, np.ndarray):
                    base = np.asarray(base)
                # TODO: This may not be a good idea
                # Create view of base array
                obj = base.view(cls) # Note: This calls obj.__array_finalize__
                # Allow conjugated to be redefined
                if conjugated is not None:
                    obj.bf.conjugated = conjugated
                    obj._update_BFarray()
            else:
                if not isinstance(base, np.ndarray):
                    # Convert base to np.ndarray
                    if dtype is not None:
                        base = np.array(base,
                                        dtype=DataType(dtype).as_numpy_dtype())
                    else:
                        base = np.array(base)
                if not isinstance(base, ndarray) and dtype is not None:
                    base = base.astype(dtype.as_numpy_dtype())
                base = ndarray(base) # View base as bf.ndarray
                if dtype is not None and base.bf.dtype != dtype:
                    raise TypeError('Unable to convert type %s to %s during '
                                    'array construction' %
                                    (base.bf.dtype, dtype))
                #base = base.view(cls
                #if dtype is not None:
                #    base = base.astype(DataType(dtype).as_numpy_dtype())
                if conjugated is None:
                    conjugated = base.bf.conjugated
                # Create copy of base array
                obj = ndarray.__new__(cls,
                                      space=space,
                                      shape=base.shape,
                                      dtype=base.bf.dtype,
                                      native=base.bf.native,
                                      conjugated=conjugated)
                copy_array(obj, base)
        else:
            # Create new array
            if dtype is None:
                dtype = 'f32' # Default dtype
            dtype = DataType(dtype)
            if native is None:
                native = True # Default byteorder
            if conjugated is None:
                conjugated = False # Default unconjugated
            if strides is None:
                #itemsize = dtype.itemsize
                itemsize_bits = dtype.itemsize_bits
                # HACK to support 'packed' arrays, by folding the last
                #   dimension of the shape into the dtype.
                # TODO: Consider using bit strides when dtype < 8 bits
                #         It's hacky, but it may be worth it
                if itemsize_bits < 8:
                    pack_factor = 8 // itemsize_bits
                    if shape[-1] % pack_factor != 0 or not len(shape):
                        raise ValueError("Array cannot be packed")
                    shape = list(shape)
                    shape[-1] //= pack_factor
                    itemsize = 1
                else:
                    itemsize = itemsize_bits // 8

                if len(shape):
                    # This magic came from http://stackoverflow.com/a/32874295
                    strides = (itemsize *
                               np.r_[1, np.cumprod(shape[::-1][:-1],
                                                   dtype=np.int64)][::-1])
                    strides = tuple(strides)
                else:
                    strides = tuple()
            nbyte = strides[0] * shape[0] if len(shape) else itemsize
            if buffer is None:
                # Allocate new buffer
                if space is None:
                    space = 'system' # Default space
                if shape is None:
                    raise ValueError('Either buffer or shape must be '
                                     'specified')
                ownbuffer = raw_malloc(nbyte, space)
                buffer = ownbuffer
            else:
                if space is None:
                    #space = _get(_bf.bfGetSpace(buffer))
                    # TODO: raw_get_space should probably return string, and needs a better name
                    space = str(Space(raw_get_space(buffer)))
            # TODO: Should move np.dtype() into as_numpy_dtype?
            dtype_np = np.dtype(dtype.as_numpy_dtype())
            if not native:
                dtype_np = dtype_np.newbyteorder()
            data_buffer = _address_as_buffer(buffer, nbyte)
            obj = np.ndarray.__new__(cls, shape, dtype_np,
                                     data_buffer, offset, strides)
            obj.bf = BFArrayInfo(space, dtype, native, conjugated, ownbuffer)
            obj._update_BFarray()
        return obj
    def __array_finalize__(self, obj):
        if obj is None:
            # Already initialized self.bf in __new__
            return
        # Initialize as view of existing array
        if isinstance(obj, ndarray):
            # Copy metadata from existing bf.ndarray
            self.bf = BFArrayInfo(obj.bf.space, obj.bf.dtype,
                                  obj.bf.native, obj.bf.conjugated)
        else:
            # Generate metadata from existing np.ndarray
            #*space      = str(Space(raw_get_space(obj.ctypes.data)))
            # Note: Assumes that any existing np.ndarray is in system space
            space      = 'system'
            #dtype      = str(DataType(obj.dtype))
            # **TODO: Decide on bf.dtype being DataType vs. string (and same for space)
            dtype      = DataType(obj.dtype)
            native     = obj.dtype.isnative
            conjugated = False
            self.bf = BFArrayInfo(space, dtype, native, conjugated)
        self._update_BFarray()
    def __del__(self):
        if hasattr(self, 'bf') and self.bf.ownbuffer:
            raw_free(self.bf.ownbuffer, self.bf.space)
    def _update_BFarray(self):
        # (Re-)cache the BFarray structure
        # Note: This must be called after any updates to self.bf.*
        self._BFarray = None
        self._BFarray = self.as_BFarray()
    def as_BFarray(self):
        # ***TODO: The caching here is broken because of shape, strides and ctypes.data
        #            How to fix?
        #*if self._BFarray is not None:
        #*    return self._BFarray
        a = _bf.BFarray()
        a.data      = self.ctypes.data
        a.space     = Space(self.bf.space).as_BFspace()
        a.dtype     = self.bf.dtype.as_BFdtype()
        a.immutable = not self.flags['WRITEABLE']
        a.ndim      = len(self.shape)
        # HACK WAR for backend not yet supporting ndim=0 (scalar arrays)
        if a.ndim == 0:
            a.ndim = 1
            a.shape[0] = 1
            a.strides[0] = self.bf.dtype.itemsize
        for d in xrange(len(self.shape)):
            a.shape[d] = self.shape[d]
        # HACK TESTING support for 'packed' arrays
        itemsize_bits = self.bf.dtype.itemsize_bits
        if itemsize_bits < 8:
            a.shape[a.ndim - 1] *= 8 // itemsize_bits
        for d in xrange(len(self.strides)):
            a.strides[d] = self.strides[d]
        a.big_endian = not self.bf.native
        a.conjugated = self.bf.conjugated
        return a
    def conj(self):
        return ndarray(self, conjugated=not self.bf.conjugated)
    def view(self, dtype=None, type_=None):
        if type_ is not None:
            dtype = type
        type_type = type(int) # HACK to form an instance of 'type' (very confusing)
        if isinstance(dtype, type_type) and issubclass(dtype, np.ndarray):
            return super(ndarray, self).view(dtype)
        else:
            # TODO: Endianness changes are not supported here
            #         Consider building byteorder into DataType
            dtype_bf = DataType(dtype)
            dtype_np = np.dtype(dtype_bf.as_numpy_dtype())
            v = super(ndarray, self).view(dtype_np)
            v.bf.dtype = dtype_bf
            v._update_BFarray()
            return v
    #def astype(self, dtype):
    #    dtype_bf = DataType(dtype)
    #    dtype_np = dtype_bf.as_numpy_dtype()
    #    # TODO: This segfaults for cuda space; need type conversion support in backend
    #    a = super(ndarray, self).astype(dtype_np)
    #    a.bf.dtype = dtype_bf
    #    return a
    def _system_accessible_copy(self):
        if space_accessible(self.bf.space, ['system']):
            return self
        else:
            return self.copy(space='system')
    def __repr__(self):
        return super(ndarray, self._system_accessible_copy()).__repr__()
    def __str__(self):
        return super(ndarray, self._system_accessible_copy()).__str__()
    def tofile(self, fid, sep="", format="%s"):
        return super(ndarray, self._system_accessible_copy()).tofile(fid, sep, format)
    def byteswap(self, inplace=False):
        if inplace:
            self.bf.native = not self.bf.native
            self._update_BFarray()
            return super(ndarray, self).byteswap(True)
        else:
            return ndarray(self).byteswap(True)
    def copy(self, space=None, order='C'):
        if order != 'C':
            raise NotImplementedError('Only order="C" is supported')
        if space is None:
            space = self.bf.space
        # Note: This makes an actual copy as long as space is not None
        return ndarray(self, space=space)
    def _key_returns_scalar(self, key):
        # Returns True if self[key] would return a scalar (i.e., not a view)
        if isinstance(key, tuple):
            if len(key) == len(self.shape):
                if all([not isinstance(k, slice) for k in key]):
                    return True
        elif not isinstance(key, slice):
            if self.shape == 1:
                return True
        return False
    def __getitem__(self, key):
        if self._key_returns_scalar(key):
            return super(ndarray, self._system_accessible_copy()).__getitem__(key)
        return super(ndarray, self).__getitem__(key)
    def __setitem__(self, key, val):
        if self._key_returns_scalar(key):
            # HACK WAR to turn key into slice to avoid scalar (non-view) result
            #   from __getitem__.
            if isinstance(key, tuple):
                key = (slice(key[0], key[0] + 1),) + key[1:]
            else:
                key = slice(key, key + 1)
        copy_array(self[key], val)
    def as_GPUArray(self, *args, **kwargs):
        from pycuda.gpuarray import GPUArray as pycuda_GPUArray
        g  = pycuda_GPUArray(shape=self.shape, dtype=self.dtype, *args, **kwargs)
        ga = asarray(g)
        copy_array(ga, self)
        return g
