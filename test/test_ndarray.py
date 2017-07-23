
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

import unittest
import numpy as np
import bifrost as bf

class NDArrayTest(unittest.TestCase):
    def setUp(self):
        self.known_vals  = [[0,1],[2,3],[4,5]]
        self.known_array = np.array(self.known_vals, dtype=np.float32)
    def test_construct(self):
        a = bf.ndarray(self.known_vals, dtype='f32')
        np.testing.assert_equal(a, self.known_array)
    def test_assign(self):
        b = bf.ndarray(shape=(3,2), dtype='f32')
        b[...] = self.known_array
        np.testing.assert_equal(b, self.known_array)
    def test_space_copy(self):
        c = bf.ndarray(self.known_vals, dtype='f32')
        c = c.copy(space='cuda').copy(space='cuda_host').copy(space='system')
        np.testing.assert_equal(c, self.known_array)
    def test_view(self):
        d = bf.ndarray(self.known_vals, dtype='f32')
        d = d.view(dtype='cf32')
        np.testing.assert_equal(d, np.array([[0 + 1j], [2 + 3j], [4 + 5j]]))
    def test_str(self):
        e = bf.ndarray(self.known_vals, dtype='f32', space='cuda')
        self.assertEqual(str(e), str(self.known_array))
    def test_repr(self):
        f = bf.ndarray(self.known_vals, dtype='f32', space='cuda')
        repr_f = repr(f)
        # Note: This chops off the class name
        repr_f = repr_f[repr_f.find('('):]
        repr_k = repr(self.known_array)
        repr_k = repr_k[repr_k.find('('):]
        # Remove whitespace (for some reason the indentation differs)
        repr_f = repr_f.replace(' ', '')
        repr_k = repr_k.replace(' ', '')
        self.assertEqual(repr_f, repr_k)
    def test_zeros_like(self):
        g = bf.ndarray(self.known_vals, dtype='f32', space='cuda')
        g = bf.zeros_like(g)
        g = g.copy('system')
        known = np.zeros_like(self.known_array)
        np.testing.assert_equal(g, known)
    def test_getitem(self):
        g = bf.ndarray(self.known_vals, space='cuda')
        np.testing.assert_equal(g[0].copy('system'),     self.known_array[0])
        np.testing.assert_equal(g[(0,)].copy('system'),  self.known_array[(0,)])
        np.testing.assert_equal(int(g[0,0]),             self.known_array[0,0])
        np.testing.assert_equal(g[:1,1:].copy('system'), self.known_array[:1,1:])
    def test_setitem(self):
        g = bf.zeros_like(self.known_vals, space='cuda')
        g[...] = self.known_vals
        np.testing.assert_equal(g.copy('system'), self.known_vals)
        g[:1,1:] = [[999]]
        np.testing.assert_equal(g.copy('system'), np.array([[0,999],[2,3],[4,5]]))
        g[0,0] = 888
        np.testing.assert_equal(g.copy('system'), np.array([[888,999],[2,3],[4,5]]))
        g[0] = [99,88]
        np.testing.assert_equal(g.copy('system'), np.array([[99,88],[2,3],[4,5]]))
        g[:,1] = [77,66,55]
        np.testing.assert_equal(g.copy('system'), np.array([[99,77],[2,66],[4,55]]))
