
# Copyright (c) 2022, The Bifrost Authors. All rights reserved.
# Copyright (c) 2022, The University of New Mexico. All rights reserved.
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
on the bifrost RAYLEIGH filter."""
# Python2 compatibility
from __future__ import division

import ctypes
import unittest
import numpy as np
from scipy.signal import lfilter, lfiltic
from bifrost.rayleigh import Rayleigh
import bifrost as bf

from bifrost.libbifrost_generated import BF_CUDA_ENABLED

MTOL = 1e-6 # Relative tolerance at the mean magnitude
RTOL = 1e-1

def compare(result, gold):
    #np.testing.assert_allclose(result, gold, RTOL, ATOL)
    # Note: We compare using an absolute tolerance equal to a fraction of the
    #         mean magnitude. This ignores large relative errors on values with
    #         magnitudes much smaller than the mean.
    absmean = np.abs(gold).mean()
    np.testing.assert_allclose(result, gold, rtol=RTOL, atol=MTOL * absmean)

@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
class TestRayleigh(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        self.shape2D = (10000, 96*2)
        self.shape3D = (10000, 48, 4)
    def test_2d(self):
        shape = self.shape2D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        
        rayleigh = Rayleigh()
        rayleigh.init(idata.shape[-1], max_flag_frac=1/idata.shape[0])
        flag, _ = rayleigh.execute(idata, odata)
        idata[5,10] = np.complex64(50+40j)
        idata[50,10] = np.complex64(50+40j)
        idata[75,10] = np.complex64(50+40j)
        flag, _ = rayleigh.execute(idata, odata)
        odata = odata.copy('system')
        
        self.assertEqual(flag.value, 1)
    def test_2d_reset(self):
        shape = self.shape2D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        
        rayleigh = Rayleigh()
        rayleigh.init(idata.shape[-1], max_flag_frac=1/idata.shape[0])
        flag, _ = rayleigh.execute(idata, odata)
        rayleigh.reset_state()
        idata[5,10] = np.complex64(50+40j)
        idata[50,10] = np.complex64(50+40j)
        idata[75,10] = np.complex64(50+40j)
        flag, _ = rayleigh.execute(idata, odata)
        odata = odata.copy('system')
        
        self.assertEqual(flag.value, 0)
    def test_3d(self):
        shape = self.shape3D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        
        rayleigh = Rayleigh()
        rayleigh.init(idata.shape[-2]*idata.shape[-1], max_flag_frac=1/idata.shape[0])
        flag, _ = rayleigh.execute(idata, odata)
        idata[5,10,0] = np.complex64(50+40j)
        idata[50,10,0] = np.complex64(50+40j)
        idata[75,10,0] = np.complex64(50+40j)
        flag, _ = rayleigh.execute(idata, odata)
        odata = odata.copy('system')
        
        self.assertEqual(flag.value, 1)
    def test_3d_reset(self):
        shape = self.shape3D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        
        rayleigh = Rayleigh()
        rayleigh.init(idata.shape[-2]*idata.shape[-1], max_flag_frac=1/idata.shape[0])
        flag, _ = rayleigh.execute(idata, odata)
        rayleigh.reset_state()
        idata[5,10,0] = np.complex64(50+40j)
        idata[50,10,0] = np.complex64(50+40j)
        idata[75,10,0] = np.complex64(50+40j)
        flag, _ = rayleigh.execute(idata, odata)
        odata = odata.copy('system')
        
        self.assertEqual(flag.value, 0)
