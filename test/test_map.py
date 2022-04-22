
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

import sys
import ctypes
import unittest
import numpy as np
import bifrost as bf
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
    
from bifrost.libbifrost_generated import BF_CUDA_ENABLED

_FIRST_TEST = True

@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
class TestMap(unittest.TestCase):
    # TODO: @classmethod; def setUpClass(kls)
    def setUp(self):
        np.random.seed(1234)
        global _FIRST_TEST
        if _FIRST_TEST:
            bf.clear_map_cache()
            _FIRST_TEST = False
    def run_simple_test(self, x, funcstr, func):
        x_orig = x
        x = bf.asarray(x, 'cuda')
        y = bf.empty_like(x)
        x.flags['WRITEABLE'] = False
        x.bf.immutable = True # TODO: Is this actually doing anything? (flags is, just not sure about bf.immutable)
        for _ in range(3):
            bf.map(funcstr, {'x': x, 'y': y})
        x = x.copy('system')
        y = y.copy('system')
        if isinstance(x_orig, bf.ndarray):
            x_orig = x
        # Note: Using func(x) is dangerous because bf.ndarray does things like
        #         lazy .conj(), which break when used as if it were np.ndarray.
        np.testing.assert_equal(y, func(x_orig))
    def run_simple_test_funcs(self, x):
        self.run_simple_test(x, "y = x+1", lambda x: x + 1)
        self.run_simple_test(x, "y = x*3", lambda x: x * 3)
        # Note: Must use "f" suffix to avoid very slow double-precision math
        self.run_simple_test(x, "y = rint(pow(x, 2.f))", lambda x: x**2)
        self.run_simple_test(x, "auto tmp = x; y = tmp*tmp", lambda x: x * x)
        self.run_simple_test(x, "y = x; y += x", lambda x: x + x)
    def test_simple_1D(self):
        n = 7919
        x = np.random.randint(256, size=n)
        self.run_simple_test_funcs(x)
    def test_simple_2D(self):
        n = 89
        x = np.random.randint(256, size=(n,n))
        self.run_simple_test_funcs(x)
    def test_simple_2D_padded(self):
        n = 89
        x = np.random.randint(256, size=(n,n))
        x = bf.asarray(x, space='cuda')
        x = x[:,1:]
        self.run_simple_test_funcs(x)
    def test_simple_3D(self):
        n = 23
        x = np.random.randint(256, size=(n,n,n))
        self.run_simple_test_funcs(x)
    def test_simple_3D_padded(self):
        n = 23
        x = np.random.randint(256, size=(n,n,n))
        x = bf.asarray(x, space='cuda')
        x = x[:,:,1:]
        self.run_simple_test_funcs(x)
        # TODO: These require bfArrayCopy to support >2D padded arrays
        #x = x[:,1:,:]
        #self.run_simple_test_funcs(x)
        #x = x[:,1:,1:]
        #self.run_simple_test_funcs(x)
    def test_broadcast(self):
        n = 89
        a = np.arange(n).astype(np.float32)
        a = bf.asarray(a, space='cuda')
        b = a[:,None]
        c = bf.empty((a.shape[0],b.shape[0]), a.dtype, 'cuda') # TODO: Need way to compute broadcast shape
        for _ in range(3):
            bf.map("c = a*b", data={'a': a, 'b': b, 'c': c})
        a = a.copy('system')
        b = b.copy('system')
        c = c.copy('system')
        np.testing.assert_equal(c, a * b)
    def test_scalar(self):
        n = 7919
        # Note: Python integer division rounds to -inf, while C rounds toward 0
        #         We avoid the problem here by using only positive values
        x = np.random.randint(1, 256, size=n)
        x = bf.asarray(x, space='cuda')
        y = bf.empty_like(x)
        for _ in range(3):
            bf.map("y = (x-m)/s", data={'x': x, 'y': y, 'm': 1, 's': 3})
        x = x.copy('system')
        y = y.copy('system')
        np.testing.assert_equal(y, (x - 1) // 3)
    def test_manydim(self):
        known_data = np.arange(3**8).reshape([3] * 8).astype(np.float32)
        a = bf.asarray(known_data, space='cuda')
        a = a[:,:,:,:,:2,:,:,:]
        b = bf.empty_like(a)
        for _ in range(3):
            bf.map("b = a+1", data={'a': a, 'b': b})
        a = a.copy('system')
        b = b.copy('system')
        np.testing.assert_equal(b, a + 1)
    def test_shift(self):
        shape = (55,66,77)
        a = np.random.randint(65536, size=shape).astype(np.int32)
        a = bf.asarray(a, space='cuda')
        b = bf.empty_like(a)
        for _ in range(3):
            bf.map("b = a(_-a.shape()/2)", data={'a': a, 'b': b})
        a = a.copy('system')
        b = b.copy('system')
        np.testing.assert_equal(b, np.fft.fftshift(a))
    def test_complex_float(self):
        n = 89
        real = np.random.randint(-127, 128, size=(n,n)).astype(np.float32)
        imag = np.random.randint(-127, 128, size=(n,n)).astype(np.float32)
        x = real + 1j * imag
        self.run_simple_test(x, "y.assign(x.imag, x.real)",
                             lambda x: x.imag + 1j * x.real)
        self.run_simple_test(x, "y = x*x.conj()", lambda x: x * x.conj())
        self.run_simple_test(x, "y = x.mag2()",   lambda x: x * x.conj())
        self.run_simple_test(x, "y = 3*x", lambda x: 3 * x)
    def test_complex_integer(self):
        n = 7919
        for in_dtype in ('ci4', 'ci8', 'ci16', 'ci32'):
            a_orig = bf.ndarray(shape=(n,), dtype=in_dtype, space='system')
            try:
                a_orig['re'] = np.random.randint(256, size=n)
                a_orig['im'] = np.random.randint(256, size=n)
            except ValueError:
                # ci4 is different
                a_orig['re_im'] = np.random.randint(256, size=n)
            for out_dtype in (in_dtype, 'cf32'):
                a = a_orig.copy(space='cuda')
                b = bf.ndarray(shape=(n,), dtype=out_dtype, space='cuda')
                bf.map('b(i) = a(i)', {'a': a, 'b': b}, shape=a.shape, axis_names=('i',))
                a = a.copy(space='system')
                try:
                    a = a['re'] + 1j*a['im']
                except ValueError:
                    # ci4 is different
                    a = np.int8(a['re_im'] & 0xF0) + 1j*np.int8((a['re_im'] & 0x0F) << 4)
                    a /= 16
                b = b.copy(space='system')
                try:
                    b = b['re'] + 1j*b['im']
                except ValueError:
                    # ci4 is different
                    b = np.int8(b['re_im'] & 0xF0) + 1j*np.int8((b['re_im'] & 0x0F) << 4)
                    b /= 16
                except IndexError:
                    # pass through cf32
                    pass
                np.testing.assert_equal(a, b)

    def test_polarisation_products(self):
        n = 89
        real = np.random.randint(-127, 128, size=(n,2)).astype(np.float32)
        imag = np.random.randint(-127, 128, size=(n,2)).astype(np.float32)
        a = real + 1j * imag
        a_orig = a
        a = bf.asarray(a, space='cuda')
        b = bf.empty_like(a)
        for _ in range(3):
            bf.map('''
            auto x = a(_,0);
            auto y = a(_,1);
            b(_,0).assign(x.mag2(), y.mag2());
            b(_,1) = x*y.conj();
            ''', shape=b.shape[:-1], data={'a': a, 'b': b})
        b = b.copy('system')
        a = a_orig
        gold = np.empty_like(a)
        def mag2(x):
            return x.real * x.real + x.imag * x.imag
        gold[...,0] = mag2(a[...,0]) + 1j * mag2(a[...,1])
        gold[...,1] = a[...,0] * a[...,1].conj()
        np.testing.assert_equal(b, gold)
    def test_explicit_indexing(self):
        shape = (55,66,77)
        a = np.random.randint(65536, size=shape).astype(np.int32)
        a = bf.asarray(a, space='cuda')
        b = bf.empty((a.shape[2],a.shape[0], a.shape[1]), a.dtype, 'cuda')
        for _ in range(3):
            bf.map("b(i,j,k) = a(j,k,i)", shape=b.shape, axis_names=('i','j','k'),
                   data={'a': a, 'b': b}, block_shape=(64,4), block_axes=('i','k'))
        a = a.copy('system')
        b = b.copy('system')
        np.testing.assert_equal(b, a.transpose([2,0,1]))
    def test_custom_shape(self):
        shape = (55,66,77)
        a = np.random.randint(65536, size=shape).astype(np.int32)
        a = bf.asarray(a, space='cuda')
        b = bf.empty((a.shape[0],a.shape[2]), a.dtype, 'cuda')
        j = 11
        for _ in range(3):
            bf.map("b(i,k) = a(i,j,k)", shape=b.shape, axis_names=('i','k'),
                   data={'a': a, 'b': b, 'j': j})
        a = a.copy('system')
        b = b.copy('system')
        np.testing.assert_equal(b, a[:,j,:])
    def test_list_cache(self):
        # TODO: would be nicer as a context manager, something like
        #       contextlib.redirect_stdout
        orig_stdout = sys.stdout
        new_stdout = StringIO()
        sys.stdout = new_stdout
        
        try:
            bf.list_map_cache()
        finally:
            sys.stdout = orig_stdout
            new_stdout.close()
