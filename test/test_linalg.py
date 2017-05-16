
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

import ctypes
import unittest
import numpy as np
from numpy import matmul as gold_matmul
from bifrost.linalg import LinAlg
import bifrost as bf

RTOL = 1e-4
ATOL = 1e-5

class TestLinAlg(unittest.TestCase):
	def setUp(self):
		self.linalg = LinAlg()
		np.random.seed(1234)
	def run_test_matmul_aa_ci8_shape(self, shape):
		shape_complex = shape[:-1] + (shape[-1]*2,)
		a8 = (np.random.random(size=shape_complex)*255).astype(np.int8)
		a_gold = a8.astype(np.float32).view(np.complex64)
		a = a8.view(bf.DataType.ci8)
		# Note: np.matmul seems to be slow and inaccurate when there are batch dims
		c_gold = np.matmul(a_gold, np.swapaxes(a_gold, -1, -2).conj())
		triu = np.triu_indices(shape[-2], 1)
		c_gold[...,triu[0],triu[1]] = 0
		a = bf.asarray(a, space='cuda')
		c = bf.zeros_like(c_gold, space='cuda')
		self.linalg.matmul(1, a, None, 0, c)
		c = c.copy('system')
		np.testing.assert_allclose(c, c_gold, RTOL, ATOL)
	def run_test_matmul_aa_dtype_shape(self, shape, dtype):
		a = ((np.random.random(size=shape))*127).astype(dtype)
		c_gold = np.matmul(a, np.swapaxes(a, -1, -2).conj())
		triu = np.triu_indices(shape[-2], 1)
		c_gold[...,triu[0],triu[1]] = 0
		a = bf.asarray(a, space='cuda')
		c = bf.zeros_like(c_gold, space='cuda')
		self.linalg.matmul(1, a, None, 0, c)
		c = c.copy('system')
		np.testing.assert_allclose(c, c_gold, RTOL, ATOL)
	def run_test_matmul_aa_dtype(self, dtype):
		self.run_test_matmul_aa_dtype_shape((11,23),         dtype)
		self.run_test_matmul_aa_dtype_shape((111,223),       dtype)
		self.run_test_matmul_aa_dtype_shape((1111,2223),     dtype)
		self.run_test_matmul_aa_dtype_shape((3,111,223),     dtype)
		self.run_test_matmul_aa_dtype_shape((5,3,111,223),   dtype)
		self.run_test_matmul_aa_dtype_shape((5,7,3,111,223), dtype)
	def test_matmul_aa_ci8(self):
		self.run_test_matmul_aa_ci8_shape((11,23))
		self.run_test_matmul_aa_ci8_shape((111,223))
		self.run_test_matmul_aa_ci8_shape((1111,2223))
		self.run_test_matmul_aa_ci8_shape((3,111,223))
		self.run_test_matmul_aa_ci8_shape((5,3,111,223))
		self.run_test_matmul_aa_ci8_shape((5,7,3,111,223))
	def test_matmul_aa_f32(self):
		self.run_test_matmul_aa_dtype(np.float32)
	def test_matmul_aa_f64(self):
		self.run_test_matmul_aa_dtype(np.float64)
	def test_matmul_aa_c32(self):
		self.run_test_matmul_aa_dtype(np.complex64)
	def test_matmul_aa_c64(self):
		self.run_test_matmul_aa_dtype(np.complex128)
