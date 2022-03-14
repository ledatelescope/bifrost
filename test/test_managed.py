
# Copyright (c) 2021-2022, The Bifrost Authors. All rights reserved.
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

import ctypes
import unittest
import numpy as np
import bifrost as bf

from bifrost.libbifrost_generated import BF_GPU_MANAGEDMEM
from bifrost.device import stream_synchronize

#
# Map
#

@unittest.skipUnless(BF_GPU_MANAGEDMEM, "requires GPU managed memory support")
class TestManagedMap(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
    def run_simple_test(self, x, funcstr, func):
        x_orig = x
        x = bf.asarray(x, 'cuda_managed')
        y = bf.empty_like(x)
        x.flags['WRITEABLE'] = False
        x.bf.immutable = True # TODO: Is this actually doing anything? (flags is, just not sure about bf.immutable)
        for _ in range(3):
            bf.map(funcstr, {'x': x, 'y': y})
            stream_synchronize()
        if isinstance(x_orig, bf.ndarray):
            x_orig = x
        # Note: Using func(x) is dangerous because bf.ndarray does things like
        #         lazy .conj(), which break when used as if it were np.ndarray.
        np.testing.assert_equal(y, func(x_orig))
    def run_simple_test_funcs(self, x):
        self.run_simple_test(x, "y = x+1", lambda x: x + 1)
        self.run_simple_test(x, "y = x*3", lambda x: x * 3)
        # Note: Must use "f" suffix to avoid very slow double-precision math
        self.run_simple_test(x, "y = rint(pow(x, 2.f))", lambda x: x**2)
        self.run_simple_test(x, "auto tmp = x; y = tmp*tmp", lambda x: x * x)
        self.run_simple_test(x, "y = x; y += x", lambda x: x + x)
    def test_simple_2D(self):
        n = 89
        x = np.random.randint(256, size=(n,n))
        self.run_simple_test_funcs(x)
    def test_simple_2D_padded(self):
        n = 89
        x = np.random.randint(256, size=(n,n))
        x = bf.asarray(x, space='cuda')
        x = x[:,1:]
        self.run_simple_test_funcs(x)

#
# FFT
#

# Note: Numpy FFTs are always double precision, which is good for this purpose
from numpy.fft import fftn as gold_fftn, ifftn as gold_ifftn
from numpy.fft import rfftn as gold_rfftn, irfftn as gold_irfftn
from bifrost.fft import Fft

MTOL = 1e-6 # Relative tolerance at the mean magnitude
RTOL = 1e-1

def compare(result, gold):
    #np.testing.assert_allclose(result, gold, RTOL, ATOL)
    # Note: We compare using an absolute tolerance equal to a fraction of the
    #         mean magnitude. This ignores large relative errors on values with
    #         magnitudes much smaller than the mean.
    absmean = np.abs(gold).mean()
    np.testing.assert_allclose(result, gold, rtol=RTOL, atol=MTOL * absmean)

@unittest.skipUnless(BF_GPU_MANAGEDMEM, "requires GPU managed memory support")
class TestManagedFFT(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        self.shape1D = (16777216,)
        self.shape2D = (2048, 2048)
        self.shape3D = (128, 128, 128)
        self.shape4D = (32, 32, 32, 32)
        # Note: Last dim must be even to avoid output alignment error
        self.shape4D_odd = (33, 31, 65, 16)
    def run_test_r2c_dtype(self, shape, axes, dtype=np.float32, scale=1., misalign=0):
        known_data = np.random.normal(size=shape).astype(np.float32)
        known_data = (known_data * scale).astype(dtype)

        # Force misaligned data
        padded_shape = shape[:-1] + (shape[-1] + misalign,)
        known_data = np.resize(known_data, padded_shape)
        idata = bf.ndarray(known_data, space='cuda_managed')
        known_data = known_data[..., misalign:]
        idata = idata[..., misalign:]

        oshape = list(shape)
        oshape[axes[-1]] = shape[axes[-1]] // 2 + 1
        odata = bf.ndarray(shape=oshape, dtype='cf32', space='cuda_managed')
        fft = Fft()
        fft.init(idata, odata, axes=axes)
        fft.execute(idata, odata)
        stream_synchronize()
        known_result = gold_rfftn(known_data.astype(np.float32) / scale, axes=axes)
        compare(odata, known_result)
    def run_test_r2c(self, shape, axes, dtype=np.float32):
        self.run_test_r2c_dtype(shape, axes, np.float32)
        # Note: Misalignment is not currently supported for fp32
        #self.run_test_r2c_dtype(shape, axes, np.float32, misalign=1)
        #self.run_test_r2c_dtype(shape, axes, np.float16) # TODO: fp16 support
        for misalign in range(4):
            self.run_test_r2c_dtype(shape, axes, np.int16, (1 << 15) - 1, misalign=misalign)
        for misalign in range(8):
            self.run_test_r2c_dtype(shape, axes, np.int8,  (1 << 7 ) - 1, misalign=misalign)
    def test_r2c_1D(self):
        self.run_test_r2c(self.shape1D, [0])
    def test_r2c_2D(self):
        self.run_test_r2c(self.shape2D, [0, 1])
    def test_r2c_3D(self):
        self.run_test_r2c(self.shape3D, [0, 1, 2])

#
# FIR
#

from scipy.signal import lfilter, lfiltic
from bifrost.fir import Fir

def compare(result, gold):
    #np.testing.assert_allclose(result, gold, RTOL, ATOL)
    # Note: We compare using an absolute tolerance equal to a fraction of the
    #         mean magnitude. This ignores large relative errors on values with
    #         magnitudes much smaller than the mean.
    absmean = np.abs(gold).mean()
    np.testing.assert_allclose(result, gold, rtol=RTOL, atol=MTOL * absmean)

@unittest.skipUnless(BF_GPU_MANAGEDMEM, "requires GPU managed memory support")
class TestManagedFIR(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        self.shape2D = (10000, 96*2)
        self.shape3D = (10000, 48, 4)
        self.coeffs = np.array(( 0.0035002, -0.0053712,  0.0090177, -0.013789,  0.0196580, 
                                -0.0264910,  0.0340400, -0.0419570,  0.049807, -0.0571210,
                                 0.0634200, -0.0682750,  0.0713370,  0.927620,  0.0713370, 
                                -0.0682750,  0.0634200, -0.0571210,  0.049807, -0.0419570, 
                                 0.0340400, -0.0264910,  0.0196580, -0.013789,  0.0090177,
                                -0.0053712, 0.0035002), dtype=np.float64)
    def test_2d_initial(self):
        shape = self.shape2D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda_managed')
        odata = bf.empty_like(idata)
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1])
        coeffs = bf.ndarray(coeffs, space='cuda_managed')
        
        fir = Fir()
        fir.init(coeffs, 1)
        fir.execute(idata, odata)
        stream_synchronize()
        
        for i in range(known_data.shape[1]):
            zf = lfiltic(self.coeffs, 1.0, 0.0)
            known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i], zi=zf)
            compare(odata[:,i], known_result)
    def test_2d_active(self):
        shape = self.shape2D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda_managed')
        odata = bf.empty_like(idata)
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1])
        coeffs = bf.ndarray(coeffs, space='cuda_managed')
        
        fir = Fir()
        fir.init(coeffs, 1)
        fir.execute(idata, odata)
        fir.execute(idata, odata)
        stream_synchronize()
        
        for i in range(known_data.shape[1]):
            zf = lfiltic(self.coeffs, 1.0, 0.0)
            known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i], zi=zf)
            known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i], zi=zf)
            compare(odata[:,i], known_result)

#
# Reduce
#

def stderr(data, axis):
    return np.sum(data, axis=axis) / np.sqrt(data.shape[axis])

NP_OPS = {
    'sum':    np.sum,
    'mean':   np.mean,
    'min':    np.min,
    'max':    np.max,
    'stderr': stderr
}

def scrunch(data, factor=2, axis=0, func=np.sum):
    if factor is None:
        factor = data.shape[axis]
    s = data.shape
    if s[axis] % factor != 0:
        raise ValueError("Scrunch factor does not divide axis size")
    s = s[:axis] + (s[axis]//factor, factor) + s[axis:][1:]
    axis = axis + 1 if axis >= 0 else axis
    return func(data.reshape(s), axis=axis)

def pwrscrunch(data, factor=2, axis=0, func=np.sum):
    if factor is None:
        factor = data.shape[axis]
    s = data.shape
    if s[axis] % factor != 0:
        raise ValueError("Scrunch factor does not divide axis size")
    s = s[:axis] + (s[axis]//factor, factor) + s[axis:][1:]
    axis = axis + 1 if axis >= 0 else axis
    return func(np.abs(data.reshape(s))**2, axis=axis)

@unittest.skipUnless(BF_GPU_MANAGEDMEM, "requires GPU managed memory support")
class TestManagedReduce(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
    def run_reduce_test(self, shape, axis, n, op='sum', dtype=np.float32):
        a = ((np.random.random(size=shape)*2-1)*127).astype(np.int8).astype(dtype)
        if op[:3] == 'pwr':
            b_gold = pwrscrunch(a.astype(np.float32), n, axis, NP_OPS[op[3:]])
        else:
            b_gold = scrunch(a.astype(np.float32), n, axis, NP_OPS[op])
        a = bf.asarray(a, space='cuda_managed')
        b = bf.empty_like(b_gold, space='cuda_managed')
        bf.reduce(a, b, op)
        stream_synchronize()
        np.testing.assert_allclose(b, b_gold)
    def test_reduce(self):
        self.run_reduce_test((3,6,5), axis=1, n=2, op='sum', dtype=np.float32)
        for shape in [(20,20,40), (20,40,60), (40,100,200)]:
            for axis in range(3):
                for n in [2, 4, 5, 10, None]:
                    for op in ['sum', 'mean', 'pwrsum', 'pwrmean']:
                        for dtype in [np.float32, np.int16, np.int8]:
                            self.run_reduce_test(shape, axis, n, op, dtype)

#
# Unpack
#

import bifrost.unpack

@unittest.skipUnless(BF_GPU_MANAGEDMEM, "requires GPU managed memory support")
class TestManagedUnpack(unittest.TestCase):
    def run_unpack_to_ci8_test(self, iarray):
        oarray = bf.ndarray(shape=iarray.shape, dtype='ci8', space='cuda_managed')
        oarray_known = bf.ndarray([[(0, 1), (2, 3)],
                                   [(4, 5), (6, 7)],
                                   [(-8, -7), (-6, -5)]],
                                  dtype='ci8')
        bf.unpack(iarray.copy(space='cuda_managed'), oarray)
        stream_synchronize()
        np.testing.assert_equal(oarray, oarray_known)
    def test_ci4_to_ci8(self):
        iarray = bf.ndarray([[(0x10,),(0x32,)],
                             [(0x54,),(0x76,)],
                             [(0x98,),(0xBA,)]],
                            dtype='ci4')
        self.run_unpack_to_ci8_test(iarray)
    def test_ci4_to_ci8_byteswap(self):
        iarray = bf.ndarray([[(0x01,),(0x23,)],
                             [(0x45,),(0x67,)],
                             [(0x89,),(0xAB,)]],
                            dtype='ci4')
        self.run_unpack_to_ci8_test(iarray.byteswap())
    def test_ci4_to_ci8_conjugate(self):
        iarray = bf.ndarray([[(0xF0,),(0xD2,)],
                             [(0xB4,),(0x96,)],
                             [(0x78,),(0x5A,)]],
                            dtype='ci4')
        self.run_unpack_to_ci8_test(iarray.conj())
    def test_ci4_to_ci8_byteswap_conjugate(self):
        iarray = bf.ndarray([[(0x0F,),(0x2D,)],
                             [(0x4B,),(0x69,)],
                             [(0x87,),(0xA5,)]],
                            dtype='ci4')
        self.run_unpack_to_ci8_test(iarray.byteswap().conj())
