
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

"""This set of unit tests check the functionality
on the bifrost FFT wrapper."""
import ctypes
import unittest
import numpy as np
# Note: Numpy FFTs are always double precision, which is good for this purpose
from numpy.fft import fftn as gold_fftn, ifftn as gold_ifftn
from numpy.fft import rfftn as gold_rfftn, irfftn as gold_irfftn
from bifrost.fft import Fft
import bifrost as bf

MTOL = 1e-6 # Relative tolerance at the mean magnitude
RTOL = 1e-1

def compare(result, gold):
    #np.testing.assert_allclose(result, gold, RTOL, ATOL)
    # Note: We compare using an absolute tolerance equal to a fraction of the
    #         mean magnitude. This ignores large relative errors on values with
    #         magnitudes much smaller than the mean.
    absmean = np.abs(gold).mean()
    np.testing.assert_allclose(result, gold, rtol=RTOL, atol=MTOL * absmean)

class TestFFT(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        self.shape1D = (16777216,)
        self.shape2D = (2048, 2048)
        self.shape3D = (128, 128, 128)
        #self.shape1D = (131072,)
        #self.shape2D = (1024,1024)
        #self.shape3D = (64,64,64)
        #self.shape4D = (64,64,64,64)
        self.shape4D = (32, 32, 32, 32)
        # Note: Last dim must be even to avoid output alignment error
        self.shape4D_odd = (33, 31, 65, 16)
    def run_test_c2c_impl(self, shape, axes, inverse=False, fftshift=False):
        shape = list(shape)
        shape[-1] *= 2 # For complex
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        fft = Fft()
        fft.init(idata, odata, axes=axes, apply_fftshift=fftshift)
        fft.execute(idata, odata, inverse)
        if inverse:
            if fftshift:
                known_data = np.fft.ifftshift(known_data, axes=axes)
            # Note: Numpy applies normalization while CUFFT does not
            norm = reduce(lambda a, b: a * b, [known_data.shape[d]
                                               for d in axes])
            known_result = gold_ifftn(known_data, axes=axes) * norm
        else:
            known_result = gold_fftn(known_data, axes=axes)
            if fftshift:
                known_result = np.fft.fftshift(known_result, axes=axes)
        x = (np.abs(odata.copy('system') - known_result) / known_result > RTOL).astype(np.int32)
        a = odata.copy('system')
        b = known_result
        compare(odata.copy('system'), known_result)
    def run_test_r2c_dtype(self, shape, axes, dtype=np.float32, scale=1., misalign=0):
        known_data = np.random.normal(size=shape).astype(np.float32)
        known_data = (known_data * scale).astype(dtype)

        # Force misaligned data
        padded_shape = shape[:-1] + (shape[-1] + misalign,)
        known_data = np.resize(known_data, padded_shape)
        idata = bf.ndarray(known_data, space='cuda')
        known_data = known_data[..., misalign:]
        idata = idata[..., misalign:]

        oshape = list(shape)
        oshape[axes[-1]] = shape[axes[-1]] // 2 + 1
        odata = bf.ndarray(shape=oshape, dtype='cf32', space='cuda')
        fft = Fft()
        fft.init(idata, odata, axes=axes)
        fft.execute(idata, odata)
        known_result = gold_rfftn(known_data.astype(np.float32) / scale, axes=axes)
        compare(odata.copy('system'), known_result)
    def run_test_r2c(self, shape, axes, dtype=np.float32):
        self.run_test_r2c_dtype(shape, axes, np.float32)
        # Note: Misalignment is not currently supported for fp32
        #self.run_test_r2c_dtype(shape, axes, np.float32, misalign=1)
        #self.run_test_r2c_dtype(shape, axes, np.float16) # TODO: fp16 support
        for misalign in xrange(4):
            self.run_test_r2c_dtype(shape, axes, np.int16, (1 << 15) - 1, misalign=misalign)
        for misalign in xrange(8):
            self.run_test_r2c_dtype(shape, axes, np.int8,  (1 << 7 ) - 1, misalign=misalign)
    def run_test_c2r_impl(self, shape, axes, fftshift=False):
        ishape = list(shape)
        oshape = list(shape)
        ishape[axes[-1]] = shape[axes[-1]] // 2 + 1
        oshape[axes[-1]] = (ishape[axes[-1]] - 1) * 2
        ishape[-1] *= 2 # For complex
        known_data = np.random.normal(size=ishape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.ndarray(shape=oshape, dtype='f32', space='cuda')
        fft = Fft()
        fft.init(idata, odata, axes=axes, apply_fftshift=fftshift)
        fft.execute(idata, odata)
        # Note: Numpy applies normalization while CUFFT does not
        norm = reduce(lambda a, b: a * b, [shape[d] for d in axes])
        if fftshift:
            known_data = np.fft.ifftshift(known_data, axes=axes)
        known_result = gold_irfftn(known_data, axes=axes) * norm
        compare(odata.copy('system'), known_result)
    def run_test_c2c(self, shape, axes):
        self.run_test_c2c_impl(shape, axes)
        self.run_test_c2c_impl(shape, axes, inverse=True)
        self.run_test_c2c_impl(shape, axes, fftshift=True)
        self.run_test_c2c_impl(shape, axes, inverse=True, fftshift=True)
    def run_test_c2r(self, shape, axes):
        self.run_test_c2r_impl(shape, axes)
        self.run_test_c2r_impl(shape, axes, fftshift=True)

    def test_1D(self):
        self.run_test_c2c(self.shape1D, [0])
    def test_1D_in_2D_dim0(self):
        self.run_test_c2c(self.shape2D, [0])
    def test_1D_in_2D_dim1(self):
        self.run_test_c2c(self.shape2D, [1])
    def test_1D_in_3D_dim0(self):
        self.run_test_c2c(self.shape3D, [0])
    def test_1D_in_3D_dim1(self):
        self.run_test_c2c(self.shape3D, [1])
    def test_1D_in_3D_dim2(self):
        self.run_test_c2c(self.shape3D, [2])

    def test_2D(self):
        self.run_test_c2c(self.shape2D, [0, 1])
    def test_2D_in_3D_dims01(self):
        self.run_test_c2c(self.shape3D, [0, 1])
    def test_2D_in_3D_dims02(self):
        self.run_test_c2c(self.shape3D, [0, 2])
    def test_2D_in_3D_dims12(self):
        self.run_test_c2c(self.shape3D, [1, 2])
    def test_2D_in_4D_dims01(self):
        self.run_test_c2c(self.shape4D, [0, 1])
    def test_2D_in_4D_dims02(self):
        self.run_test_c2c(self.shape4D, [0, 2])
        self.run_test_c2c(self.shape4D_odd, [0, 2])
    def test_2D_in_4D_dims03(self):
        self.run_test_c2c(self.shape4D, [0, 3])
    def test_2D_in_4D_dims12(self):
        self.run_test_c2c(self.shape4D, [1, 2])
    def test_2D_in_4D_dims13(self):
        self.run_test_c2c(self.shape4D, [1, 3])
    def test_2D_in_4D_dims23(self):
        self.run_test_c2c(self.shape4D, [2, 3])

    def test_3D(self):
        self.run_test_c2c(self.shape3D, [0, 1, 2])
    def test_3D_in_4D_dims012(self):
        self.run_test_c2c(self.shape4D, [0, 1, 2])
    def test_3D_in_4D_dims013(self):
        self.run_test_c2c(self.shape4D, [0, 1, 3])
    def test_3D_in_4D_dims023(self):
        self.run_test_c2c(self.shape4D, [0, 2, 3])
    def test_3D_in_4D_dims123(self):
        self.run_test_c2c(self.shape4D, [1, 2, 3])

    def test_r2c_1D(self):
        self.run_test_r2c(self.shape1D, [0])
    def test_r2c_2D(self):
        self.run_test_r2c(self.shape2D, [0, 1])
    def test_r2c_3D(self):
        self.run_test_r2c(self.shape3D, [0, 1, 2])

    def test_c2r_1D(self):
        self.run_test_c2r(self.shape1D, [0])
    def test_c2r_2D(self):
        self.run_test_c2r(self.shape2D, [0, 1])
    def test_c2r_3D(self):
        self.run_test_c2r(self.shape3D, [0, 1, 2])

    def test_r2c_2D_in_3D_dims01(self):
        self.run_test_r2c(self.shape3D, [0, 1])
    def test_r2c_2D_in_3D_dims02(self):
        self.run_test_r2c(self.shape3D, [0, 2])
    def test_r2c_2D_in_3D_dims12(self):
        self.run_test_r2c(self.shape3D, [1, 2])

    def test_r2c_2D_in_4D_dims01(self):
        self.run_test_r2c(self.shape4D, [0, 1])
    def test_r2c_2D_in_4D_dims02(self):
        self.run_test_r2c(self.shape4D, [0, 2])
        self.run_test_r2c(self.shape4D_odd, [0, 2])
    def test_r2c_2D_in_4D_dims03(self):
        self.run_test_r2c(self.shape4D, [0, 3])
    def test_r2c_2D_in_4D_dims12(self):
        self.run_test_r2c(self.shape4D, [1, 2])
    def test_r2c_2D_in_4D_dims13(self):
        self.run_test_r2c(self.shape4D, [1, 3])
    def test_r2c_2D_in_4D_dims23(self):
        self.run_test_r2c(self.shape4D, [2, 3])

    def test_c2r_2D_in_4D_dims01(self):
        self.run_test_c2r(self.shape4D, [0, 1])
    def test_c2r_2D_in_4D_dims02(self):
        self.run_test_c2r(self.shape4D, [0, 2])
        self.run_test_c2r(self.shape4D_odd, [0, 2])
    def test_c2r_2D_in_4D_dims03(self):
        self.run_test_c2r(self.shape4D, [0, 3])
    def test_c2r_2D_in_4D_dims12(self):
        self.run_test_c2r(self.shape4D, [1, 2])
    def test_c2r_2D_in_4D_dims13(self):
        self.run_test_c2r(self.shape4D, [1, 3])
    def test_c2r_2D_in_4D_dims23(self):
        self.run_test_c2r(self.shape4D, [2, 3])
