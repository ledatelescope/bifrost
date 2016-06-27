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

from bifrost.GPUArray import GPUArray
from bifrost.ring import Ring
from bifrost.fft import fft
from bifrost.libbifrost import _bf, _string2space
import numpy as np
import ctypes


BF_MAX_DIM = 3

class BFarray(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("space", _bf.BFspace),
        ("dtype", ctypes.c_uint),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_ulong*BF_MAX_DIM),
        ("strides", ctypes.c_ulong*BF_MAX_DIM)]

np.random.seed(4)
a = GPUArray(shape=(10), dtype=np.float32)
a.set(np.arange(10))
data = ctypes.cast(a.ctypes.data, ctypes.c_void_p)
space = _string2space('cuda')
c = (ctypes.c_ulong*BF_MAX_DIM)(*[10,0,0])
d = (ctypes.c_ulong*BF_MAX_DIM)(*[4*8,0,0])
myarray = BFarray(data,space,1,1,c,d)
print _bf.BFarray
#mybfarray = ctypes.cast(myarray,_bf.BFarray)
print fft(myarray, myarray)

