
# Copyright (c) 2016-2020, The Bifrost Authors. All rights reserved.
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
i:    signed integer
u:  unsigned integer
f:  floating point
ci: complex   signed integer
cf: complex floating pointer

i4:   4-bit signed integer
f16:  16-bit floating point
ci4:  4+4-bit complex signed integer
cf32: 32+32-bit complex floating point
"""

# Python2 compatibility
from __future__ import division, absolute_import
import sys
string_types = (str,)
if sys.version_info < (3,):
    range = xrange
    string_types = (basestring,)
    
from bifrost.libbifrost import _bf
import numpy as np

# Custom dtypes to represent additional complex types
# Note: These can be constructed using tuples
#   E.g., np.ndarray([(0x10,), (0x32,)], dtype=ci4) # Special case
#         np.ndarray([  (0,1),   (2,3)], dtype=ci8)
#         np.ndarray([  (0,1),   (2,3)], dtype=ci16)
ci4  = np.dtype([('re_im', np.int8)])
ci8  = np.dtype([('re', np.int8),    ('im', np.int8)])
ci16 = np.dtype([('re', np.int16),   ('im', np.int16)])
ci32 = np.dtype([('re', np.int32),   ('im', np.int32)])
ci64 = np.dtype([('re', np.int64),   ('im', np.int64)])
cf16 = np.dtype([('re', np.float16), ('im', np.float16)])

TYPEMAP = {
    'i':  { 1: _bf.BF_DTYPE_I1,   2: _bf.BF_DTYPE_I2,
            4: _bf.BF_DTYPE_I4,   8: _bf.BF_DTYPE_I8,
           16: _bf.BF_DTYPE_I16, 32: _bf.BF_DTYPE_I32,
           64: _bf.BF_DTYPE_I64},
    'u':  { 1: _bf.BF_DTYPE_U1,   2: _bf.BF_DTYPE_U2,
            4: _bf.BF_DTYPE_U4,   8: _bf.BF_DTYPE_U8,
           16: _bf.BF_DTYPE_U16, 32: _bf.BF_DTYPE_U32,
           64: _bf.BF_DTYPE_U64},
    'f':  {16: _bf.BF_DTYPE_F16,  32: _bf.BF_DTYPE_F32,
           64: _bf.BF_DTYPE_F64, 128: _bf.BF_DTYPE_F128},
    'ci': { 1: _bf.BF_DTYPE_CI1,   2: _bf.BF_DTYPE_CI2,
            4: _bf.BF_DTYPE_CI4,   8: _bf.BF_DTYPE_CI8,
           16: _bf.BF_DTYPE_CI16, 32: _bf.BF_DTYPE_CI32,
           64: _bf.BF_DTYPE_CI64},
    'cf': {16: _bf.BF_DTYPE_CF16,  32: _bf.BF_DTYPE_CF32,
           64: _bf.BF_DTYPE_CF64, 128: _bf.BF_DTYPE_CF128}
}
KINDMAP = {
    _bf.BF_DTYPE_INT_TYPE:   'i',
    _bf.BF_DTYPE_UINT_TYPE:  'u',
    _bf.BF_DTYPE_FLOAT_TYPE: 'f'
}
NUMPY_TYPEMAP = {
    'i':  {  8: np.int8,  16: np.int16,
             32: np.int32, 64: np.int64},
    'u':  {  8: np.uint8,  16: np.uint16,
             32: np.uint32, 64: np.uint64},
    'f':  {16: np.float16,  32: np.float32,
           64: np.float64, 128: np.float128},
    # HACK: These are just types that match the storage size;
    #         they should not be used for computation.
    # HACK TESTING to support 'packed' arrays
    #   (TODO: Do same for 'i' and 'u' if happy with this)
    'ci': { 1: np.int8,  2: np.int8,
            4: ci4,      8: ci8,
            16: ci16,    32: ci32,
            64: ci64},
    # HACK: cf16 used as WAR for missing np.complex32
    'cf': {16: cf16,           32: np.complex64,
           64: np.complex128, 128: np.complex256}
}

def is_vector_structure(dtype):
    if dtype.names is None:
        return False
    ndim = len(dtype.names)
    vector_field_names = tuple('f%i' % i for i in range(ndim))
    return (dtype.kind == 'V' and
            dtype.names == vector_field_names and
            all([dtype[i] == dtype[0] for i in range(1, ndim)]))

class DataType(object):
    # Note: Default of None results in default Numpy type (np.float)
    def __init__(self, t=None):
        if isinstance(t, string_types):
            for i, char in enumerate(t):
                if char.isdigit():
                    break
            self._kind =     t[:i]
            self._nbit = int(t[i:])
            self._veclen = 1 # TODO: Consider supporting this as part of string
        elif isinstance(t, _bf.BFdtype): # Note: This is actually just a c_int
            t = int(t)
            self._nbit = t & BF_DTYPE_NBIT_BITS
            is_complex = bool(t & _bf.BF_DTYPE_COMPLEX_BIT)
            self._kind = KINDMAP[t & _bf.BF_DTYPE_TYPE_BITS]
            if is_complex:
                self._kind = 'c' + self._kind
            self._veclen = 1 + ((t & _bf.BF_DTYPE_VECTOR_BITS)
                                >> _bf.BF_DTYPE_VECTOR_BIT0)
        elif isinstance(t, DataType):
            self._nbit = t._nbit
            self._kind = t._kind
            self._veclen = t._veclen
        elif isinstance(t, tuple):
            self._kind, self._nbit, self._veclen = t
        else:
            t = np.dtype(t) # Raises TypeError if t is invalid
            ndim = getattr(t, 'ndim', len(t.shape))
            if ndim == 0:
                self._veclen = 1
            elif ndim == 1:
                self._veclen = t.shape[0]
                t = t.base
            else:
                raise TypeError("Unsupported Numpy dtype: " + str(t))
            self._nbit = t.itemsize * 8
            if t.kind not in set(['i', 'u', 'f', 'c', 'V', 'b']):
                raise TypeError('Unsupported data type: %s' % str(t))
            if is_vector_structure(t): # Field structure representing vector
                self._veclen = len(t.names)
                t = t[0]
            self._kind = t.kind
            if t.kind == 'c':
                self._nbit //= 2   # Bifrost convention is nbit per real component
                self._kind = 'cf' # Numpy only supports floating-point complex types
            elif t.kind == 'V': # WAR to support custom integer complex types
                self._nbit //= 2
                if t in [ci4, ci8, ci16, ci32, ci64]:
                    self._kind = 'ci'
                elif t in [cf16]:
                    self._kind = 'cf'
                else:
                    raise TypeError('Unsupported data type: %s' % str(t))
            elif t.kind == 'b':
                # Note: Represents booleans as uint8 inside Bifrost
                self._kind = 'u'
    def __eq__(self, other):
        return (self._kind == other._kind and
                self._nbit == other._nbit and
                self._veclen == other._veclen)
    def __ne__(self, other):
        return not (self == other)
    def as_BFdtype(self):
        base = TYPEMAP[self._kind][self._nbit]
        return base | ((self._veclen - 1) << _bf.BF_DTYPE_VECTOR_BIT0)
    def as_numpy_dtype(self):
        base = np.dtype(NUMPY_TYPEMAP[self._kind][self._nbit])
        if self._veclen == 1:
            return base
        else:
            #return np.dtype((base, self._veclen))
            # WAR for vector types not working as expected when passed to
            #   np.view (the shape gets merged into the ndarray's shape instead
            #   of remaining part of the dtype). We return a structure type
            #   with fields representing a vector instead.
            return np.dtype(','.join((str(base),)*self._veclen))
    def __str__(self):
        if self._veclen == 1:
            return '%s%i' % (self._kind, self._nbit)
        else:
            return '%s%i[%i]' % (self._kind, self._nbit, self._veclen)
    @property
    def is_complex(self):
        return self._kind[0] == 'c'
    @property
    def is_real(self):
        return not self.is_complex
    @property
    def is_signed(self):
        return 'i' in self._kind or 'f' in self._kind
    @property
    def is_floating_point(self):
        return 'f' in self._kind
    @property
    def is_integer(self):
        return 'i' in self._kind or 'u' in self._kind
    def as_floating_point(self):
        """Returns the smallest floating-point type that can represent all
        values that self can.
        """
        if self.is_floating_point:
            return self
        kind = 'cf' if self.is_complex else 'f'
        nbit = 32 if self._nbit <= 24 else 64
        return DataType((kind, nbit, self._veclen))
    def as_integer(self, nbit=None):
        if nbit is None:
            nbit = self._nbit
        kind = self._kind
        if self.is_floating_point:
            kind = kind.replace('f', 'i')
        return DataType((kind, nbit, self._veclen))
    def as_real(self):
        if self.is_complex:
            return DataType((self._kind[1:], self._nbit, self._veclen))
        else:
            return self
    def as_complex(self):
        if self.is_complex:
            return self
        else:
            return DataType(('c' + self._kind, self._nbit, self._veclen))
    def as_nbit(self, nbit):
        return DataType((self._kind, nbit, self._veclen))
    def as_vector(self, veclen):
        return DataType((self._kind, self._nbit, veclen))
    @property
    def itemsize_bits(self):
        return self._nbit * (1 + self.is_complex) * self._veclen
    @property
    def itemsize(self):
        item_nbit = self.itemsize_bits
        if item_nbit < 8:
            raise ValueError('itemsize is undefined when nbit < 8')
        return item_nbit // 8
