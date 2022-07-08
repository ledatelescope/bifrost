
# Copyright (c) 2022, The Bifrost Authors. All rights reserved.
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
on the bifrost Graph wrapper."""

import unittest
import numpy as np
import bifrost as bf
from bifrost.ndarray import copy_array
from bifrost.fft import Fft

from bifrost.libbifrost_generated import BF_CUDA_ENABLED

@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
class TestGraph(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        self.shape2D = (128, 2048)
    def test_graph(self):
        n = 7917
        x = np.random.randint(256, size=n)
        
        x_orig = bf.asarray(x)
        
        # TODO: We can't make a call to stream_synchronize() inside of a graph
        # capture so we can't directly use bifrost.ndarray.copy_array when the
        # two arrays arenin different spaces.
        x = bf.asarray(x_orig, space='cuda')
        
        graph = bf.device.Graph()
        for i in range(10):
            with graph:
                if graph.created:
                    raise bf.device.GraphCreatedError
                    
                bf.map('x = x + 3', {'x': x})
            bf.device.stream_synchronize()
            
        x = x.copy('system')
        np.testing.assert_equal(x, x_orig+10*3)
    def test_graph_copy(self):
        n = 7917
        x = np.random.randint(256, size=n)
        
        x_orig = bf.asarray(x)
        
        # TODO: We can't make a call to stream_synchronize() inside of a graph
        # capture so we can't directly use bifrost.ndarray.copy_array when the
        # two arrays arenin different spaces.
        x = bf.asarray(x_orig, space='cuda')
        bf.map('x = x + 1', {'x': x})
        
        graph = bf.device.Graph()
        for i in range(10):
            with graph:
                if graph.created:
                    raise bf.device.GraphCreatedError
                    
                bf.map('x = x + 1', {'x': x})
                graph.copy_array(x, x_orig)
                bf.map('x = x + 3', {'x': x})
            bf.device.stream_synchronize()
            
        x = x.copy('system')
        np.testing.assert_equal(x, x_orig+3)
    def test_graph_fft(self):
        shape = list(self.shape2D)
        shape[-1] *= 2 # For complex
        known_data = np.random.normal(size=shape).astype(np.float32).view(np.complex64)
        x = bf.ndarray(known_data, space='cuda')
        y = bf.empty_like(x)
        
        graph = bf.device.Graph()
        for i in range(10):
            with graph:
                if graph.created:
                    raise bf.device.GraphCreatedError
                    
                try:
                    f.execute(x, y, inverse=False)
                except NameError:
                    f = Fft()
                    f.init(x, y, axes=0)
                    f.execute(x, y, inverse=False)
            bf.device.stream_synchronize()
            
        y = y.copy('system')
        y_gold = np.fft.fft2(known_data, axes=[0,])
        absmean = np.abs(y_gold).mean()
        np.testing.assert_allclose(y, y_gold, rtol=1e-1, atol=1e-6 * absmean)
