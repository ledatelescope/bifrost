
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

from __future__ import division

import unittest
import numpy as np
import bifrost as bf

from bifrost.libbifrost_generated import BF_CUDA_ENABLED

#import time

def stderr(data, axis):
    return np.sum(data, axis=axis) / np.sqrt(data.shape[axis])

NP_OPS = {
    'sum':    np.sum,
    'mean':   np.mean,
    'min':    np.min,
    'max':    np.max,
    'stderr': stderr
}

def scrunch(data, factor=2, axis=0, func=np.sum):
    if factor is None:
        factor = data.shape[axis]
    s = data.shape
    if s[axis] % factor != 0:
        raise ValueError("Scrunch factor does not divide axis size")
    s = s[:axis] + (s[axis]//factor, factor) + s[axis:][1:]
    axis = axis + 1 if axis >= 0 else axis
    return func(data.reshape(s), axis=axis)

def pwrscrunch(data, factor=2, axis=0, func=np.sum):
    if factor is None:
        factor = data.shape[axis]
    s = data.shape
    if s[axis] % factor != 0:
        raise ValueError("Scrunch factor does not divide axis size")
    s = s[:axis] + (s[axis]//factor, factor) + s[axis:][1:]
    axis = axis + 1 if axis >= 0 else axis
    return func(np.abs(data.reshape(s))**2, axis=axis)

@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
class ReduceTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
    def run_reduce_test(self, shape, axis, n, op='sum', dtype=np.float32):
        a = ((np.random.random(size=shape)*2-1)*127).astype(np.int8).astype(dtype)
        if op[:3] == 'pwr':
            b_gold = pwrscrunch(a.astype(np.float32), n, axis, NP_OPS[op[3:]])
        else:
            b_gold = scrunch(a.astype(np.float32), n, axis, NP_OPS[op])
        a = bf.asarray(a, space='cuda')
        b = bf.empty_like(b_gold, space='cuda')
        bf.reduce(a, b, op)
        b = b.copy('system')
        np.testing.assert_allclose(b, b_gold)
    def run_reduce_slice_test(self, shape, axis, n, op='sum', dtype=np.float32):
        if n is None:
            return None
        a = ((np.random.random(size=shape)*2-1)*127).astype(np.int8).astype(dtype)
        if axis == 0:
            a_slice = a[1:((a.shape[0]-1)//n-1)*n+1,...]
        elif axis == 1:
            a_slice = a[:,1:((a.shape[1]-1)//n-1)*n+1,:]
        else:
            a_slice = a[...,1:((a.shape[-1]-1)//n-1)*n+1]
        if a_slice.shape[0] == 0 or a_slice.shape[1] == 0 or a_slice.shape[-1] == 0:
            return None
        if op[:3] == 'pwr':
            b_gold = pwrscrunch(a_slice.astype(np.float32), n, axis, NP_OPS[op[3:]])
        else:
            b_gold = scrunch(a_slice.astype(np.float32), n, axis, NP_OPS[op])
        a = bf.ndarray(a, space='cuda')
        if axis == 0:
            a_slice = a[1:((a.shape[0]-1)//n-1)*n+1,...]
        elif axis == 1:
            a_slice = a[:,1:((a.shape[1]-1)//n-1)*n+1,:]
        else:
            a_slice = a[...,1:((a.shape[-1]-1)//n-1)*n+1]
        b = bf.empty_like(b_gold, space='cuda')
        bf.reduce(a_slice, b, op)
        b = b.copy('system')
        np.testing.assert_allclose(b, b_gold)
    def test_reduce(self):
        self.run_reduce_test((3,6,5), axis=1, n=2, op='sum', dtype=np.float32)
        for shape in [(20,20,40), (20,40,60), (40,100,200)]:
            for axis in range(3):
                for n in [2, 4, 5, 10, None]:
                    for op in ['sum', 'mean', 'pwrsum', 'pwrmean']:#, 'min', 'max', 'stderr']:
                        for dtype in [np.float32, np.int16, np.int8]:
                            #print shape, axis, n, op, dtype
                            self.run_reduce_test(shape, axis, n, op, dtype)
                            self.run_reduce_slice_test(shape, axis, n, op, dtype)
    def test_reduce_pow2(self):
        for shape in [(16,32,64), (16,64,256), (256,64,16)]:#, (256, 256, 512)]:
            for axis in range(3):
                for n in [2, 4, 8, 16, None]:
                    for op in ['sum', 'mean', 'pwrsum', 'pwrmean']:#, 'min', 'max', 'stderr']:
                        for dtype in [np.float32, np.int16, np.int8]:
                            #print shape, axis, n, op, dtype
                            self.run_reduce_test(shape, axis, n, op, dtype)
                            self.run_reduce_slice_test(shape, axis, n, op, dtype)
    
    def run_complex_reduce_test(self, shape, axis, n, op='sum', dtype=np.complex64):
        a = ((np.random.random(size=shape)*2-1)*127).astype(np.int8).astype(dtype) \
            + 1j*((np.random.random(size=shape)*2-1)*127).astype(np.int8).astype(dtype)
        if op[:3] == 'pwr':
            b_gold = pwrscrunch(a.astype(np.complex64), n, axis, NP_OPS[op[3:]]).astype(np.float32)
        else:
            b_gold = scrunch(a.astype(np.complex64), n, axis, NP_OPS[op])
        a = bf.asarray(a, space='cuda')
        b = bf.empty_like(b_gold, space='cuda')
        bf.reduce(a, b, op)
        b = b.copy('system')
        np.testing.assert_allclose(b, b_gold, rtol=1e-3 if op[:3] == 'pwr' else 1e-7)
    def run_complex_reduce_slice_test(self, shape, axis, n, op='sum', dtype=np.float32):
        if n is None:
            return None
        a = ((np.random.random(size=shape)*2-1)*127).astype(np.int8).astype(dtype) \
            + 1j*((np.random.random(size=shape)*2-1)*127).astype(np.int8).astype(dtype)
        if axis == 0:
            a_slice = a[1:((a.shape[0]-1)//n-1)*n+1,...]
        elif axis == 1:
            a_slice = a[:,1:((a.shape[1]-1)//n-1)*n+1,:]
        else:
            a_slice = a[...,1:((a.shape[-1]-1)//n-1)*n+1]
        if a_slice.shape[0] == 0 or a_slice.shape[1] == 0 or a_slice.shape[-1] == 0:
            return None
        if op[:3] == 'pwr':
            b_gold = pwrscrunch(a_slice.astype(np.complex64), n, axis, NP_OPS[op[3:]])
        else:
            b_gold = scrunch(a_slice.astype(np.complex64), n, axis, NP_OPS[op])
        a = bf.ndarray(a, space='cuda')
        if axis == 0:
            a_slice = a[1:((a.shape[0]-1)//n-1)*n+1,...]
        elif axis == 1:
            a_slice = a[:,1:((a.shape[1]-1)//n-1)*n+1,:]
        else:
            a_slice = a[...,1:((a.shape[-1]-1)//n-1)*n+1]
        b = bf.empty_like(b_gold, space='cuda')
        bf.reduce(a_slice, b, op)
        b = b.copy('system')
        np.testing.assert_allclose(b, b_gold, rtol=1e-3 if op[:3] == 'pwr' else 1e-7)
    def test_complex_reduce(self):
        self.run_complex_reduce_test((3,6,5), axis=1, n=2, op='pwrsum', dtype=np.complex64)
        for shape in [(20,20,40), (20,40,60), (40,100,200)]:
            for axis in range(3):
                for n in [2, 4, 5, 10, None]:
                    for op in ['sum', 'mean', 'pwrsum', 'pwrmean']:#, 'min', 'max', 'stderr']:
                        for dtype in [np.complex64,]:
                            #print shape, axis, n, op, dtype
                            self.run_complex_reduce_test(shape, axis, n, op, dtype)
                            self.run_complex_reduce_slice_test(shape, axis, n, op, dtype)
    def test_complex_reduce_pow2(self):
        for shape in [(16,32,64), (16,64,256), (256,64,16)]:#, (256, 256, 512)]:
            for axis in range(3):
                for n in [2, 4, 8, 16, None]:
                    for op in ['sum', 'mean', 'pwrsum', 'pwrmean']:#, 'min', 'max', 'stderr']:
                        for dtype in [np.complex64,]:
                            #print shape, axis, n, op, dtype
                            self.run_complex_reduce_test(shape, axis, n, op, dtype)
                            self.run_complex_reduce_slice_test(shape, axis, n, op, dtype)
