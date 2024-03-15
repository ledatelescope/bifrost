
# Copyright (c) 2016-2022, The Bifrost Authors. All rights reserved.
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
import ctypes

from bifrost.libbifrost_generated import BF_CUDA_ENABLED
from bifrost.DataType import DataType

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
    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
    def test_space_copy(self):
        c = bf.ndarray(self.known_vals, dtype='f32')
        c = c.copy(space='cuda').copy(space='cuda_host').copy(space='system')
        np.testing.assert_equal(c, self.known_array)
    def run_contiguous_copy(self, space='system'):
        a = np.random.rand(2,3,4,5)
        a = a.astype(np.float64)
        b = a.transpose(0,3,2,1).copy()
        c = bf.zeros(a.shape, dtype=a.dtype, space='system')
        c[...] = a
        c = c.copy(space=space)
        d = c.transpose(0,3,2,1).copy(space='system')
        # Use ctypes to directly access the memory
        b_data = ctypes.cast(b.ctypes.data, ctypes.POINTER(ctypes.c_double))
        b_data = np.array([b_data[i] for i in range(b.size)])
        d_data = ctypes.cast(d.ctypes.data, ctypes.POINTER(ctypes.c_double))
        d_data = np.array([d_data[i] for i in range(d.size)])
        np.testing.assert_equal(d_data, b_data)
    def test_contiguous_copy(self):
        self.run_contiguous_copy()
    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
    def test_space_contiguous_copy(self):
        self.run_contiguous_copy(space='cuda')
    def run_slice_copy(self, space='system'):
        a = np.random.rand(2,3,4,5)
        a = a.astype(np.float64)
        b = a[:,1:,:,:].copy()
        c = bf.zeros(a.shape, dtype=a.dtype, space='system')
        c[...] = a
        c = c.copy(space=space)
        d = c[:,1:,:,:].copy(space='system')
        # Use ctypes to directly access the memory
        b_data = ctypes.cast(b.ctypes.data, ctypes.POINTER(ctypes.c_double))
        b_data = np.array([b_data[i] for i in range(b.size)])
        d_data = ctypes.cast(d.ctypes.data, ctypes.POINTER(ctypes.c_double))
        d_data = np.array([d_data[i] for i in range(d.size)])
        np.testing.assert_equal(d_data, b_data)
    def test_slice_copy(self):
        self.run_slice_copy()
    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
    def test_space_slice_copy(self):
        self.run_slice_copy(space='cuda')
    def run_contiguous_slice_copy(self, space='system'):
        a = np.random.rand(2,3,4,5)
        a = a.astype(np.float64)
        b = a.transpose(0,3,2,1)[:,1:,:,:].copy()
        c = bf.zeros(a.shape, dtype=a.dtype, space='system')
        c[...] = a
        c = c.copy(space=space)
        d = c.transpose(0,3,2,1)[:,1:,:,:].copy(space='system')
        # Use ctypes to directly access the memory
        b_data = ctypes.cast(b.ctypes.data, ctypes.POINTER(ctypes.c_double))
        b_data = np.array([b_data[i] for i in range(b.size)])
        d_data = ctypes.cast(d.ctypes.data, ctypes.POINTER(ctypes.c_double))
        d_data = np.array([d_data[i] for i in range(d.size)])
        np.testing.assert_equal(d_data, b_data)
    def test_contiguous_slice_copy(self):
        self.run_contiguous_slice_copy()
    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
    def test_space_contiguous_slice_copy(self):
        self.run_contiguous_slice_copy(space='cuda')
    def test_view(self):
        d = bf.ndarray(self.known_vals, dtype='f32')
        d = d.view(dtype='cf32')
        np.testing.assert_equal(d, np.array([[0 + 1j], [2 + 3j], [4 + 5j]]))
    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
    def test_str(self):
        e = bf.ndarray(self.known_vals, dtype='f32', space='cuda')
        self.assertEqual(str(e), str(self.known_array))
    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
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
    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
    def test_zeros_like(self):
        g = bf.ndarray(self.known_vals, dtype='f32', space='cuda')
        g = bf.zeros_like(g)
        g = g.copy('system')
        known = np.zeros_like(self.known_array)
        np.testing.assert_equal(g, known)
    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
    def test_getitem(self):
        g = bf.ndarray(self.known_vals, space='cuda')
        np.testing.assert_equal(g[0].copy('system'),     self.known_array[0])
        np.testing.assert_equal(g[(0,)].copy('system'),  self.known_array[(0,)])
        np.testing.assert_equal(int(g[0,0]),             self.known_array[0,0])
        np.testing.assert_equal(g[:1,1:].copy('system'), self.known_array[:1,1:])
    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
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
    def run_type_conversion(self, space='system'):
        # Real
        for dtype_in in (np.int8, np.int16, np.int32, np.float32, np.float64):
            a = np.array(self.known_vals, dtype=dtype_in)
            c = bf.ndarray(a, dtype=dtype_in, space=space)
            for dtype in ('i8', 'i16', 'i32', 'i64', 'f64', 'ci8', 'ci16', 'ci32', 'cf32', 'cf64'):
                np_dtype = DataType(dtype).as_numpy_dtype()
                try:
                    ## Catch for the complex integer types
                    len(np_dtype)
                    b = np.zeros(a.shape, dtype=np_dtype)
                    b['re'] = a
                except (IndexError, TypeError):
                    b = a.astype(np_dtype)
                d = c.astype(dtype)
                d = d.copy(space='system')
                np.testing.assert_equal(b, d)
        # Complex
        for dtype_in,dtype_in_cmplx in zip((np.float32,np.float64), ('cf32', 'cf64')):
            a = np.array(self.known_vals, dtype=dtype_in)
            a = np.stack([a,a[::-1]], axis=0)
            a = a.view(np.complex64)
            c = bf.ndarray(a, dtype=dtype_in_cmplx, space=space)
            for dtype in ('ci8', 'ci16', 'ci32', 'cf32', 'cf64', 'i8', 'i16', 'i32', 'i64', 'f64'):
                np_dtype = DataType(dtype).as_numpy_dtype()
                try:
                    ## Catch for the complex integer types
                    len(np_dtype)
                    b = np.zeros(a.shape, dtype=np_dtype)
                    b['re'] = a.real
                    b['im'] = a.imag
                except (IndexError, TypeError):
                    b = a.astype(np_dtype)
                d = c.astype(dtype)
                d = d.copy(space='system')
                np.testing.assert_equal(b, d)
    def test_type_conversion(self):
        self.run_type_conversion()
    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
    def test_space_type_conversion(self):
        self.run_type_conversion(space='cuda')
    def test_BFarray(self):
        """ Test ndarray.as_BFarray() roundtrip """
        a = bf.ndarray(np.arange(100), dtype='i32')
        aa = a.as_BFarray()
        b = bf.ndarray(aa)
        np.testing.assert_equal(a, b)

        a = bf.ndarray(np.arange(100), dtype='cf32')
        aa = a.as_BFarray()
        b = bf.ndarray(aa)
        np.testing.assert_equal(a, b)
