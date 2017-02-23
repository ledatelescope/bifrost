
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

"""This set of unit tests check the functionality
on the bifrost FFT wrapper, both in device memory and out"""
import ctypes
import unittest
import numpy as np
# Note: Numpy FFTs are always double precision, which is good for this purpose
from numpy.fft import fftn as gold_fftn, ifftn as gold_ifftn
from numpy.fft import rfftn as gold_rfftn, irfftn as gold_irfftn
from bifrost.fft import Fft
import bifrost as bf

# TODO: These tolerances are way too high, but only a tiny fraction of the
#         result values have such large errors. Need a better way to quantify.
RTOL = 1e-1
ATOL = 1e-1

class TestFFT(unittest.TestCase):
	def setUp(self):
		np.random.seed(1234)
		self.shape1D = (16777216,)
		self.shape2D = (4096, 4096)
		self.shape3D = (256,256,256)
		#self.shape1D = (131072,)
		#self.shape2D = (1024,1024)
		#self.shape3D = (64,64,64)
		#self.shape4D = (64,64,64,64)
		self.shape4D = (32,32,32,32)
	def run_test_c2c_impl(self, shape, axes, inverse=False):
		shape = list(shape)
		shape[-1] *= 2 # For complex
		known_data = np.random.uniform(size=shape).astype(np.float32).view(np.complex64)
		idata = bf.ndarray(known_data, space='cuda')
		odata = bf.empty_like(idata)
		fft = Fft()
		fft.init(idata, odata, axes=axes)
		fft.execute(idata, odata, inverse)
		if inverse:
			# Note: Numpy applies normalization while CUFFT does not
			norm = reduce(lambda a,b: a*b, [known_data.shape[d] for d in axes])
			known_result = gold_ifftn(known_data, axes=axes) * norm
		else:
			known_result = gold_fftn(known_data, axes=axes)
		np.testing.assert_allclose(odata.copy('system'), known_result, RTOL, ATOL)
	def run_test_r2c(self, shape, axes):
		known_data = np.random.uniform(size=shape).astype(np.float32)
		idata = bf.ndarray(known_data, space='cuda')
		oshape = list(shape)
		oshape[axes[-1]] = shape[axes[-1]] // 2 + 1
		odata = bf.ndarray(shape=oshape, dtype='cf32', space='cuda')
		fft = Fft()
		fft.init(idata, odata, axes=axes)
		fft.execute(idata, odata)
		known_result = gold_rfftn(known_data, axes=axes)
		np.testing.assert_allclose(odata.copy('system'), known_result, RTOL, ATOL)
	def run_test_c2r(self, shape, axes):
		ishape = list(shape)
		ishape[axes[-1]] = shape[axes[-1]] // 2 + 1
		ishape[-1] *= 2 # For complex
		known_data = np.random.uniform(size=ishape).astype(np.float32).view(np.complex64)
		idata = bf.ndarray(known_data, space='cuda')
		odata = bf.ndarray(shape=shape, dtype='f32', space='cuda')
		fft = Fft()
		fft.init(idata, odata, axes=axes)
		fft.execute(idata, odata)
		# Note: Numpy applies normalization while CUFFT does not
		norm = reduce(lambda a,b: a*b, [shape[d] for d in axes])
		known_result = gold_irfftn(known_data, axes=axes) * norm
		np.testing.assert_allclose(odata.copy('system'), known_result, RTOL, ATOL)
	def run_test_c2c(self, shape, axes):
		self.run_test_c2c_impl(shape, axes)
		self.run_test_c2c_impl(shape, axes, inverse=True)
	
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
		self.run_test_c2c(self.shape2D, [0,1])
	def test_2D_in_3D_dims01(self):
		self.run_test_c2c(self.shape3D, [0,1])
	def test_2D_in_3D_dims02(self):
		self.run_test_c2c(self.shape3D, [0,2])
	def test_2D_in_3D_dims12(self):
		self.run_test_c2c(self.shape3D, [1,2])
	def test_2D_in_4D_dims01(self):
		self.run_test_c2c(self.shape4D, [0,1])
	def test_2D_in_4D_dims02(self):
		self.run_test_c2c(self.shape4D, [0,2])
	def test_2D_in_4D_dims03(self):
		self.run_test_c2c(self.shape4D, [0,3])
	def test_2D_in_4D_dims12(self):
		self.run_test_c2c(self.shape4D, [1,2])
	def test_2D_in_4D_dims13(self):
		self.run_test_c2c(self.shape4D, [1,3])
	def test_2D_in_4D_dims23(self):
		self.run_test_c2c(self.shape4D, [2,3])
	
	def test_3D(self):
		self.run_test_c2c(self.shape3D, [0,1,2])
	def test_3D_in_4D_dims012(self):
		self.run_test_c2c(self.shape4D, [0,1,2])
	def test_3D_in_4D_dims013(self):
		self.run_test_c2c(self.shape4D, [0,1,3])
	def test_3D_in_4D_dims023(self):
		self.run_test_c2c(self.shape4D, [0,2,3])
	def test_3D_in_4D_dims123(self):
		self.run_test_c2c(self.shape4D, [1,2,3])
	
	def test_r2c_1D(self):
		self.run_test_r2c(self.shape1D, [0])
	def test_r2c_2D(self):
		self.run_test_r2c(self.shape2D, [0,1])
	def test_r2c_3D(self):
		self.run_test_r2c(self.shape3D, [0,1,2])
	
	def test_c2r_1D(self):
		self.run_test_c2r(self.shape1D, [0])
	def test_c2r_2D(self):
		self.run_test_c2r(self.shape2D, [0,1])
	def test_c2r_3D(self):
		self.run_test_c2r(self.shape3D, [0,1,2])
	
	def test_r2c_2D_in_3D_dims01(self):
		self.run_test_r2c(self.shape3D, [0,1])
	def test_r2c_2D_in_3D_dims02(self):
		self.run_test_r2c(self.shape3D, [0,2])
	def test_r2c_2D_in_3D_dims12(self):
		self.run_test_r2c(self.shape3D, [1,2])
	
	def test_r2c_2D_in_4D_dims01(self):
		self.run_test_r2c(self.shape4D, [0,1])
	def test_r2c_2D_in_4D_dims02(self):
		self.run_test_r2c(self.shape4D, [0,2])
	def test_r2c_2D_in_4D_dims03(self):
		self.run_test_r2c(self.shape4D, [0,3])
	def test_r2c_2D_in_4D_dims12(self):
		self.run_test_r2c(self.shape4D, [1,2])
	def test_r2c_2D_in_4D_dims13(self):
		self.run_test_r2c(self.shape4D, [1,3])
	def test_r2c_2D_in_4D_dims23(self):
		self.run_test_r2c(self.shape4D, [2,3])
	
	def test_c2r_2D_in_4D_dims01(self):
		self.run_test_c2r(self.shape4D, [0,1])
	def test_c2r_2D_in_4D_dims02(self):
		self.run_test_c2r(self.shape4D, [0,2])
	def test_c2r_2D_in_4D_dims03(self):
		self.run_test_c2r(self.shape4D, [0,3])
	def test_c2r_2D_in_4D_dims12(self):
		self.run_test_c2r(self.shape4D, [1,2])
	def test_c2r_2D_in_4D_dims13(self):
		self.run_test_c2r(self.shape4D, [1,3])
	def test_c2r_2D_in_4D_dims23(self):
		self.run_test_c2r(self.shape4D, [2,3])
