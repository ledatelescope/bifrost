
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

i:    signed integer
u:  unsigned integer
f:  floating point
ci: complex   signed integer
cu: complex unsigned integer
cf: complex floating pointer

i4:   4-bit signed integer
f16:  16-bit floating point
ci4:  4+4-bit complex signed integer
cf32: 32+32-bit complex floating point

"""

from libbifrost import _bf
import numpy as np

def split_name_nbit(dtype_str):
    """Splits a dtype string into (name, nbit)"""
    for i, char in enumerate(dtype_str):
        if char.isdigit():
            break
    name =     dtype_str[:i]
    nbit = int(dtype_str[i:])
    return name, nbit

# Custom dtypes to represent additional complex types
ci8  = np.dtype([('re', np.int8),    ('im', np.int8)])
ci16 = np.dtype([('re', np.int16),   ('im', np.int16)])
ci32 = np.dtype([('re', np.int32),   ('im', np.int32)])
cf16 = np.dtype([('re', np.float16), ('im', np.float16)])
def to_complex64(q):
    real_type = q.dtype['re']
    return q.view(real_type).astype(np.float32).view(np.complex64)
def from_complex64(f, dtype):
    real_type = dtype['re']
    return f.view(np.float32).astype(real_type).view(dtype)

def numpy2bifrost(dtype):
    if   dtype == np.int8:       return _bf.BF_DTYPE_I8
    elif dtype == np.int16:      return _bf.BF_DTYPE_I16
    elif dtype == np.int32:      return _bf.BF_DTYPE_I32
    elif dtype == np.uint8:      return _bf.BF_DTYPE_U8
    elif dtype == np.uint16:     return _bf.BF_DTYPE_U16
    elif dtype == np.uint32:     return _bf.BF_DTYPE_U32
    elif dtype == np.float16:    return _bf.BF_DTYPE_F16
    elif dtype == np.float32:    return _bf.BF_DTYPE_F32
    elif dtype == np.float64:    return _bf.BF_DTYPE_F64
    elif dtype == np.float128:   return _bf.BF_DTYPE_F128
    elif dtype == ci8:           return _bf.BF_DTYPE_CI8
    elif dtype == ci16:          return _bf.BF_DTYPE_CI16
    elif dtype == ci32:          return _bf.BF_DTYPE_CI32
    elif dtype == cf16:          return _bf.BF_DTYPE_CF16
    elif dtype == np.complex64:  return _bf.BF_DTYPE_CF32
    elif dtype == np.complex128: return _bf.BF_DTYPE_CF64
    elif dtype == np.complex256: return _bf.BF_DTYPE_CF128
    else: raise ValueError("Unsupported dtype: " + str(dtype))

def name_nbit2numpy(name, nbit):
    if   name == 'i':
        if   nbit == 8:   return np.int8
        elif nbit == 16:  return np.int16
        elif nbit == 32:  return np.int32
        elif nbit == 64:  return np.int64
        else: raise TypeError("Invalid signed integer type size: %i" % nbit)
    elif name == 'u':
        if   nbit == 8:   return np.uint8
        elif nbit == 16:  return np.uint16
        elif nbit == 32:  return np.uint32
        elif nbit == 64:  return np.uint64
        else: raise TypeError("Invalid unsigned integer type size: %i" % nbit)
    elif name == 'f':
        if   nbit == 16:  return np.float16
        elif nbit == 32:  return np.float32
        elif nbit == 64:  return np.float64
        elif nbit == 128: return np.float128
        else: raise TypeError("Invalid floating-point type size: %i" % nbit)
    elif name == 'ci':
        if   nbit == 8:   return ci8
        elif nbit == 16:  return ci16
        elif nbit == 32:  return ci32
    # elif name in set(['ci', 'cu']):
        # Note: This gives integer types in place of proper complex types
        # return name_nbit2numpy(name[1:], nbit*2)
    elif name == 'cf':
        if   nbit == 16:  return cf16
        elif nbit == 32:  return np.complex64
        elif nbit == 64:  return np.complex128
        elif nbit == 128: return np.complex256
        else: raise TypeError("Invalid complex floating-point type size: %i" %
                              nbit)
    else:
        raise TypeError("Invalid type name: " + name)
def string2numpy(dtype_str):
    return name_nbit2numpy(*split_name_nbit(dtype_str))

def numpy2string(dtype):
    if   dtype == np.int8:       return 'i8'
    elif dtype == np.int16:      return 'i16'
    elif dtype == np.int32:      return 'i32'
    elif dtype == np.int64:      return 'i64'
    elif dtype == np.uint8:      return 'u8'
    elif dtype == np.uint16:     return 'u16'
    elif dtype == np.uint32:     return 'u32'
    elif dtype == np.uint64:     return 'u64'
    elif dtype == np.float16:    return 'f16'
    elif dtype == np.float32:    return 'f32'
    elif dtype == np.float64:    return 'f64'
    elif dtype == np.float128:   return 'f128'
    elif dtype == np.complex64:  return 'cf32'
    elif dtype == np.complex128: return 'cf64'
    elif dtype == np.complex256: return 'cf128'
    else: raise TypeError("Unsupported dtype: " + str(dtype))
