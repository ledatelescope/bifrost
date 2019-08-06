
# Copyright (c) 2017, The Bifrost Authors. All rights reserved.
# Copyright (c) 2017, The University of New Mexico. All rights reserved.
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
on the bifrost FIR filter."""
import ctypes
import unittest
import numpy as np
from scipy.signal import lfilter, lfiltic
from bifrost.fir import Fir
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

class TestFIR(unittest.TestCase):
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
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1])
        coeffs = bf.ndarray(coeffs, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 1)
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            zf = lfiltic(self.coeffs, 1.0, 0.0)
            known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i], zi=zf)
            compare(odata[:,i], known_result)
            
    def test_3d_initial(self):
        shape = self.shape3D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1]*idata.shape[2], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1],idata.shape[2])
        coeffs = bf.ndarray(coeffs, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 1)
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            for j in xrange(known_data.shape[2]):
                zf = lfiltic(self.coeffs, 1.0, 0.0)
                known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i,j], zi=zf)
                compare(odata[:,i,j], known_result)
                
    def test_2d_and_3d(self):
        shape = self.shape3D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1]*idata.shape[2], axis=1)
        coeffs = bf.ndarray(coeffs, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 1)
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            for j in xrange(known_data.shape[2]):
                zf = lfiltic(self.coeffs, 1.0, 0.0)
                known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i,j], zi=zf)
                compare(odata[:,i,j], known_result)
                
    def test_3d_and_2d(self):
        shape = self.shape2D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1]/2, 2)
        coeffs = bf.ndarray(coeffs, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 1)
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            zf = lfiltic(self.coeffs, 1.0, 0.0)
            known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i], zi=zf)
            compare(odata[:,i], known_result)
            
    def test_2d_active(self):
        shape = self.shape2D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1])
        coeffs = bf.ndarray(coeffs, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 1)
        fir.execute(idata, odata)
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            zf = lfiltic(self.coeffs, 1.0, 0.0)
            known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i], zi=zf)
            known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i], zi=zf)
            compare(odata[:,i], known_result)
            
    def test_3d_active(self):
        shape = self.shape3D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1]*idata.shape[2], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1],idata.shape[2])
        coeffs = bf.ndarray(coeffs, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 1)
        fir.execute(idata, odata)
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            for j in xrange(known_data.shape[2]):
                zf = lfiltic(self.coeffs, 1.0, 0.0)
                known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i,j], zi=zf)
                known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i,j], zi=zf)
                compare(odata[:,i,j], known_result)
                
    def test_2d_decimate_initial(self):
        shape = self.shape2D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty((idata.shape[0]/2, idata.shape[1]), dtype=idata.dtype, space='cuda')
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1])
        coeffs = bf.ndarray(coeffs, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 2)
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            zf = lfiltic(self.coeffs, 1.0, 0.0)
            known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i], zi=zf)
            known_result = known_result[0::2]
            compare(odata[:,i], known_result)
            
    def test_3d_decimate_initial(self):
        shape = self.shape3D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty((idata.shape[0]/2, idata.shape[1], idata.shape[2]), dtype=idata.dtype, space='cuda')
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1]*idata.shape[2], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1],idata.shape[2])
        coeffs = bf.ndarray(coeffs, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 2)
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            for j in xrange(known_data.shape[2]):
                zf = lfiltic(self.coeffs, 1.0, 0.0)
                known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i,j], zi=zf)
                known_result = known_result[0::2]
                compare(odata[:,i,j], known_result)
                
    def test_2d_decimate_active(self):
        shape = self.shape2D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty((idata.shape[0]/2, idata.shape[1]), dtype=idata.dtype, space='cuda')
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1])
        coeffs = bf.ndarray(coeffs, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 2)
        fir.execute(idata, odata)
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            zf = lfiltic(self.coeffs, 1.0, 0.0)
            known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i], zi=zf)
            known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i], zi=zf)
            known_result = known_result[0::2]
            compare(odata[:,i], known_result)
            
    def test_3d_decimate_active(self):
        shape = self.shape3D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty((idata.shape[0]/2, idata.shape[1], idata.shape[2]), dtype=idata.dtype, space='cuda')
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1]*idata.shape[2], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1],idata.shape[2])
        coeffs = bf.ndarray(coeffs, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 2)
        fir.execute(idata, odata)
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            for j in xrange(known_data.shape[2]):
                zf = lfiltic(self.coeffs, 1.0, 0.0)
                known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i,j], zi=zf)
                known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i,j], zi=zf)
                known_result = known_result[0::2]
                compare(odata[:,i,j], known_result)
                
    def test_2d_update_coeffs(self):
        shape = self.shape2D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1])
        coeffs2 = coeffs*2.0
        coeffs = bf.ndarray(coeffs, space='cuda')
        coeffs2 = bf.ndarray(coeffs2, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 1)
        fir.execute(idata, odata)
        fir.set_coeffs(coeffs2)
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            zf = lfiltic(self.coeffs, 1.0, 0.0)
            known_result, zf = lfilter(self.coeffs*2.0, 1.0, known_data[:,i], zi=zf)
            compare(odata[:,i], known_result)
            
    def test_3d_update_coeffs(self):
        shape = self.shape3D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1]*idata.shape[2], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1],idata.shape[2])
        coeffs2 = coeffs*2.0
        coeffs = bf.ndarray(coeffs, space='cuda')
        coeffs2 = bf.ndarray(coeffs2, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 1)
        fir.execute(idata, odata)
        fir.set_coeffs(coeffs2)
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            for j in xrange(known_data.shape[2]):
                zf = lfiltic(self.coeffs, 1.0, 0.0)
                known_result, zf = lfilter(self.coeffs*2.0, 1.0, known_data[:,i,j], zi=zf)
                compare(odata[:,i,j], known_result)
                
    def test_2d_reset_state(self):
        shape = self.shape2D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1])
        coeffs = bf.ndarray(coeffs, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 1)
        fir.execute(idata, odata)
        fir.reset_state()
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            zf = lfiltic(self.coeffs, 1.0, 0.0)
            known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i], zi=zf)
            compare(odata[:,i], known_result)
            
    def test_3d_reset_state(self):
        shape = self.shape3D
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        idata = bf.ndarray(known_data, space='cuda')
        odata = bf.empty_like(idata)
        coeffs = self.coeffs*1.0
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, idata.shape[1]*idata.shape[2], axis=1)
        coeffs.shape = (coeffs.shape[0],idata.shape[1],idata.shape[2])
        coeffs2 = coeffs*2.0
        coeffs = bf.ndarray(coeffs, space='cuda')
        coeffs2 = bf.ndarray(coeffs2, space='cuda')
        
        fir = Fir()
        fir.init(coeffs, 1)
        fir.execute(idata, odata)
        fir.reset_state()
        fir.execute(idata, odata)
        odata = odata.copy('system')
        
        for i in xrange(known_data.shape[1]):
            for j in xrange(known_data.shape[2]):
                zf = lfiltic(self.coeffs, 1.0, 0.0)
                known_result, zf = lfilter(self.coeffs, 1.0, known_data[:,i,j], zi=zf)
                compare(odata[:,i,j], known_result)
                