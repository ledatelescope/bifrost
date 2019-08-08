
# Copyright (c) 2019, The Bifrost Authors. All rights reserved.
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

from __future__ import print_function

import unittest
import numpy
import bifrost
from bifrost.fft_shift import fft_shift_2d

GRID_SIZE = 16

class FftShiftTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(1337)
        data = numpy.random.randn(10,GRID_SIZE,GRID_SIZE)*10
        self.data_even = data + 5j*numpy.random.randn(*data.shape)
        data = numpy.random.randn(10,GRID_SIZE+1,GRID_SIZE+1)*10
        self.data_odd = data + 5j*numpy.random.randn(*data.shape)
        
    def run_test(self, data, axes=(-2,-1)):
        # Reference from numpy.fft.fftshift
        gold = numpy.fft.fftshift(data, axes=axes)
        
        # Copy to the GPU, run fft_shift_2d, and copy back
        data = bifrost.ndarray(data)
        data = data.copy(space='cuda')
        out = fft_shift_2d(data, data.shape[axes[1]], reduce(lambda x,y:x*y, data.shape[:-2]))
        out = out.copy(space='system')
        
        # Check
        numpy.testing.assert_equal(out, gold)
        
    def _prepare_data(self, data, dtype):
        if not numpy.issubdtype(numpy.complex, dtype):
            data = data.real
        data = data.astype(dtype)
        return data
        
    def test_i8_even(self):
        data = self._prepare_data(self.data_even, numpy.int8)
        self.run_test(data)
    def test_i8_odd(self):
        data = self._prepare_data(self.data_odd, numpy.int8)
        self.run_test(data)
    def test_i16(self):
        data = self._prepare_data(self.data_even, numpy.int16)
        self.run_test(data)
    def test_i16_odd(self):
        data = self._prepare_data(self.data_odd, numpy.int16)
        self.run_test(data)
    def test_f32_even(self):
        data = self._prepare_data(self.data_even, numpy.float32)
        self.run_test(data)
    def test_f32_odd(self):
        data = self._prepare_data(self.data_odd, numpy.float32)
        self.run_test(data)
    def test_cf32_even(self):
        data = self._prepare_data(self.data_even, numpy.complex64)
        self.run_test(data)
    def test_cf32_odd(self):
        data = self._prepare_data(self.data_odd, numpy.complex64)
        self.run_test(data)
    def test_cf64_even(self):
        data = self._prepare_data(self.data_even, numpy.complex128)
        self.run_test(data)
    def test_cf64_odd(self):
        data = self._prepare_data(self.data_odd, numpy.complex128)
        self.run_test(data)
