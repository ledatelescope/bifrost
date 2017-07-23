
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

import ctypes
import unittest
import numpy as np
from numpy import matmul as gold_matmul
from bifrost.linalg import LinAlg
import bifrost as bf

RTOL = 1e-4
ATOL = 1e-5

def H(c):
    return np.swapaxes(c, -1, -2).conj()

class TestLinAlg(unittest.TestCase):
    def setUp(self):
        self.linalg = LinAlg()
        np.random.seed(1234)
    def run_test_matmul_aa_ci8_shape(self, shape, transpose=False):
        shape_complex = shape[:-1] + (shape[-1] * 2,)
        a8 = (np.random.random(size=shape_complex) * 255).astype(np.int8)
        a_gold = a8.astype(np.float32).view(np.complex64)
        if transpose:
            a_gold = H(a_gold)
        # Note: np.matmul seems to be slow and inaccurate when there are batch dims
        c_gold = np.matmul(a_gold, np.swapaxes(a_gold, -1, -2).conj())
        triu = np.triu_indices(shape[-2] if not transpose else shape[-1], 1)
        c_gold[..., triu[0], triu[1]] = 0
        a = a8.view(bf.DataType.ci8)
        a = bf.asarray(a, space='cuda')
        if transpose:
            a = H(a)
        c = bf.zeros_like(c_gold, space='cuda')
        self.linalg.matmul(1, a, None, 0, c)
        c = c.copy('system')
        np.testing.assert_allclose(c, c_gold, RTOL, ATOL)
    def run_test_matmul_aa_dtype_shape(self, shape, dtype, axes=None):
        a = ((np.random.random(size=shape)) * 127).astype(dtype)
        if axes is None:
            axes = range(len(shape))
        aa = a.transpose(axes)
        c_gold = np.matmul(aa, np.swapaxes(aa, -1, -2).conj())
        triu = np.triu_indices(shape[axes[-2]], 1)
        c_gold[..., triu[0], triu[1]] = 0
        a = bf.asarray(a, space='cuda')
        aa = a.transpose(axes)
        c = bf.zeros_like(c_gold, space='cuda')
        self.linalg.matmul(1, aa, None, 0, c)
        c = c.copy('system')
        np.testing.assert_allclose(c, c_gold, RTOL, ATOL)
    def run_test_matmul_ab_ci8_shape(self, shape, k, transpose=False):
        ashape_complex = shape[:-2] + (shape[-2], k * 2)
        bshape_complex = shape[:-2] + (k, shape[-1] * 2)
        a8 = (np.random.random(size=ashape_complex) * 255).astype(np.int8)
        b8 = (np.random.random(size=bshape_complex) * 255).astype(np.int8)
        a_gold = a8.astype(np.float32).view(np.complex64)
        b_gold = b8.astype(np.float32).view(np.complex64)
        if transpose:
            a_gold, b_gold = H(b_gold), H(a_gold)
        c_gold = np.matmul(a_gold, b_gold)
        a = a8.view(bf.DataType.ci8)
        b = b8.view(bf.DataType.ci8)
        a = bf.asarray(a, space='cuda')
        b = bf.asarray(b, space='cuda')
        if transpose:
            a, b = H(b), H(a)
        c = bf.zeros_like(c_gold, space='cuda')
        self.linalg.matmul(1, a, b, 0, c)
        c = c.copy('system')
        np.testing.assert_allclose(c, c_gold, RTOL, ATOL)
    def run_test_matmul_ab_dtype_shape(self, shape, k, dtype,
                                       axes_a=None, axes_b=None, transpose=False):
        ashape = shape[:-2] + (shape[-2], k)
        bshape = shape[:-2] + (k, shape[-1])
        a = ((np.random.random(size=ashape)) * 127).astype(dtype)
        b = ((np.random.random(size=bshape)) * 127).astype(dtype)
        if axes_a is None:
            axes_a = range(len(ashape))
        if axes_b is None:
            axes_b = range(len(bshape))
        aa = a.transpose(axes_a)
        bb = b.transpose(axes_b)
        if transpose:
            aa, bb = H(bb), H(aa)
        c_gold = np.matmul(aa, bb)
        a = bf.asarray(a, space='cuda')
        b = bf.asarray(b, space='cuda')
        aa = a.transpose(axes_a)
        bb = b.transpose(axes_b)
        if transpose:
            aa, bb = H(bb), H(aa)
        c = bf.zeros_like(c_gold, space='cuda')
        self.linalg.matmul(1, aa, bb, 0, c)
        c = c.copy('system')
        np.testing.assert_allclose(c, c_gold, RTOL, ATOL)
    def run_test_matmul_aa_dtype(self, dtype):
        self.run_test_matmul_aa_dtype_shape((11,23),         dtype)
        self.run_test_matmul_aa_dtype_shape((11,23),         dtype, [1,0])
        self.run_test_matmul_aa_dtype_shape((111,223),       dtype)
        self.run_test_matmul_aa_dtype_shape((111,223),       dtype, [1,0])
        self.run_test_matmul_aa_dtype_shape((1111,2223),     dtype)
        self.run_test_matmul_aa_dtype_shape((3,111,223),     dtype)
        self.run_test_matmul_aa_dtype_shape((3,111,223),     dtype, [0,2,1])
        self.run_test_matmul_aa_dtype_shape((3,111,223),     dtype, [1,2,0])
        self.run_test_matmul_aa_dtype_shape((3,111,223),     dtype, [1,0,2])
        # Note: The fastest dim can't be a batch dim, so these aren't supported
        #self.run_test_matmul_aa_dtype_shape((3,111,223),     dtype, [2,0,1])
        #self.run_test_matmul_aa_dtype_shape((3,111,223),     dtype, [2,1,0])
        self.run_test_matmul_aa_dtype_shape((5,3,111,57),   dtype)
        self.run_test_matmul_aa_dtype_shape((5,3,111,57),   dtype, [0,1,3,2])
        self.run_test_matmul_aa_dtype_shape((5,3,111,57),   dtype, [1,0,2,3])
        self.run_test_matmul_aa_dtype_shape((5,3,111,57),   dtype, [1,0,3,2])
        self.run_test_matmul_aa_dtype_shape((5,3,111,57),   dtype, [1,2,3,0])
        self.run_test_matmul_aa_dtype_shape((5,3,111,57),   dtype, [1,2,0,3])
        self.run_test_matmul_aa_dtype_shape((5,3,111,57),   dtype, [2,1,0,3])
        self.run_test_matmul_aa_dtype_shape((5,3,111,57),   dtype, [2,1,3,0])
        self.run_test_matmul_aa_dtype_shape((5,3,111,57),   dtype, [2,0,3,1])
        self.run_test_matmul_aa_dtype_shape((5,3,111,57),   dtype, [2,0,1,3])
        self.run_test_matmul_aa_dtype_shape((5,7,3,111,223), dtype)
    def run_test_matmul_ab_dtype_transpose(self, dtype, transpose):
        self.run_test_matmul_ab_dtype_shape((11,23),       7, dtype, transpose=transpose)
        self.run_test_matmul_ab_dtype_shape((11,23),      11, dtype, transpose=transpose)
        self.run_test_matmul_ab_dtype_shape((11,23),      23, dtype, transpose=transpose)
        self.run_test_matmul_ab_dtype_shape((11,11),      11, dtype, transpose=transpose)
        self.run_test_matmul_ab_dtype_shape((111,223),    77, dtype, transpose=transpose)
        self.run_test_matmul_ab_dtype_shape((111,2223),  777, dtype, transpose=transpose)
        self.run_test_matmul_ab_dtype_shape((3,111,223),  77, dtype, transpose=transpose)
    def run_test_matmul_ab_dtype(self, dtype):
        self.run_test_matmul_ab_dtype_transpose(dtype, False)
        self.run_test_matmul_ab_dtype_transpose(dtype, True)
    def run_test_matmul_aa_ci8_transpose(self, transpose):
        self.run_test_matmul_aa_ci8_shape((11,23),         transpose=transpose)
        self.run_test_matmul_aa_ci8_shape((111,223),       transpose=transpose)
        self.run_test_matmul_aa_ci8_shape((1111,2223),     transpose=transpose)
        self.run_test_matmul_aa_ci8_shape((3,111,223),     transpose=transpose)
        self.run_test_matmul_aa_ci8_shape((5,3,111,223),   transpose=transpose)
        self.run_test_matmul_aa_ci8_shape((5,7,3,111,223), transpose=transpose)
    def test_matmul_aa_ci8(self):
        self.run_test_matmul_aa_ci8_transpose(False)
        self.run_test_matmul_aa_ci8_transpose(True)
    def run_test_matmul_ab_ci8_transpose(self, transpose):
        self.run_test_matmul_ab_ci8_shape((11,23),       7777, transpose=transpose)
        self.run_test_matmul_ab_ci8_shape((111,223),      777, transpose=transpose)
        self.run_test_matmul_ab_ci8_shape((1111,2223),     77, transpose=transpose)
        self.run_test_matmul_ab_ci8_shape((3,111,223),     77, transpose=transpose)
        self.run_test_matmul_ab_ci8_shape((5,3,111,223),   77, transpose=transpose)
        self.run_test_matmul_ab_ci8_shape((5,7,3,111,223), 77, transpose=transpose)
    def test_matmul_ab_ci8(self):
        self.run_test_matmul_ab_ci8_transpose(False)
        self.run_test_matmul_ab_ci8_transpose(True)
    def test_matmul_aa_f32(self):
        self.run_test_matmul_aa_dtype(np.float32)
    def test_matmul_aa_f64(self):
        self.run_test_matmul_aa_dtype(np.float64)
    def test_matmul_aa_cf32(self):
        self.run_test_matmul_aa_dtype(np.complex64)
    def test_matmul_aa_cf64(self):
        self.run_test_matmul_aa_dtype(np.complex128)
    def test_matmul_ab_f32(self):
        self.run_test_matmul_ab_dtype(np.float32)
    def test_matmul_ab_f64(self):
        self.run_test_matmul_ab_dtype(np.float64)
    def test_matmul_ab_cf32(self):
        self.run_test_matmul_ab_dtype(np.complex64)
    def test_matmul_ab_cf64(self):
        self.run_test_matmul_ab_dtype(np.complex128)
