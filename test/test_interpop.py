
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

import unittest
import numpy as np
import bifrost as bf

from bifrost.libbifrost_generated import BF_CUDA_ENABLED

try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import pycuda.gpuarray
    import pycuda.driver
    HAVE_PYCUDA = True
except ImportError:
    HAVE_PYCUDA = False

@unittest.skipUnless(BF_CUDA_ENABLED and HAVE_CUPY, "requires GPU support and cupy")
class TestCuPy(unittest.TestCase):
    @staticmethod
    def create_data():
        data = np.random.rand(100,1000)
        return data.astype(np.float32)
        
    def test_as_cupy(self):
        data = self.create_data()
        
        bf_data = bf.ndarray(data, space='cuda')
        cp_data = bf_data.as_cupy()
        np_data = cp.asnumpy(cp_data)
        np.testing.assert_allclose(np_data, data)
    def test_from_cupy(self):
        data = self.create_data()
        
        cp_data = cp.asarray(data)
        bf_data = bf.ndarray(cp_data)
        np_data = bf_data.copy(space='system')
        np.testing.assert_allclose(np_data, data)
        
    def test_stream(self):
        data = self.create_data()
        
        with cp.cuda.Stream() as stream:
            with bf.device.ExternalStream(stream):
                self.assertEqual(bf.device.get_stream(), stream.ptr)
                
                bf_data = bf.ndarray(data, space='cuda')
                bf.map('a = a + 2', {'a': bf_data})
                cp_data = bf_data.as_cupy()
                cp_data *= 4
                np_data = cp.asnumpy(cp_data)
        np.testing.assert_allclose(np_data, (data+2)*4)
    def test_external_stream(self):
        data = self.create_data()
        
        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            self.assertEqual(cp.cuda.get_current_stream().ptr, stream)
            
            bf_data = bf.ndarray(data, space='cuda')
            bf.map('a = a + 2', {'a': bf_data})
            cp_data = bf_data.as_cupy()
            cp_data *= 4
            np_data = cp.asnumpy(cp_data)
        np.testing.assert_allclose(np_data, (data+2)*4)

@unittest.skipUnless(BF_CUDA_ENABLED and HAVE_PYCUDA, "requires GPU support and cupy")
class TestPyCUDA(unittest.TestCase):
    @staticmethod
    def create_data():
        data = np.random.rand(100,1000)
        return data.astype(np.float32)
        
    def test_as_gpuarray(self):
        data = self.create_data()
        
        bf_data = bf.ndarray(data, space='cuda')
        pc_data = bf_data.as_GPUArray()
        np_data = pc_data.get()
        np.testing.assert_allclose(np_data, data)
    def test_from_gpuarray(self):
        data = self.create_data()
        
        pc_data = pycuda.gpuarray.to_gpu(data)
        bf_data = bf.ndarray(pc_data)
        np_data = bf_data.copy(space='system')
        np.testing.assert_allclose(np_data, data)
        
    def test_stream(self):
        data = self.create_data()
        
        stream = pycuda.driver.Stream()
        with bf.device.ExternalStream(stream):
            self.assertEqual(bf.device.get_stream(), stream.handle)
            
            bf_data = bf.ndarray(data, space='cuda')
            bf.map('a = a + 2', {'a': bf_data})
            cp_data = bf_data.as_cupy()
            cp_data *= 4
            np_data = cp.asnumpy(cp_data)
        np.testing.assert_allclose(np_data, (data+2)*4)
