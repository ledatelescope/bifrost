
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
import bifrost.transpose
from itertools import permutations

class TransposeTest(unittest.TestCase):
    def run_simple_test(self, axes, dtype, shape):
        n = reduce(lambda a,b:a*b, shape)
        idata = (np.arange(n).reshape(shape) % 251).astype(dtype)
        odata_gold = idata.transpose(axes)
        iarray = bf.ndarray(idata, space='cuda')
        oarray = bf.empty_like(iarray.transpose(axes))
        bf.transpose.transpose(oarray, iarray, axes)
        oarray = oarray.copy('system')
        np.testing.assert_array_equal(oarray, odata_gold)

    def run_simple_test_shmoo(self, dtype):
        for perm in permutations(xrange(3)):
            for shape in [(23,37,51),
                          (100, 200, 2),
                          (2, 200, 100)]:
                self.run_simple_test(perm, dtype, shape)
    def test_1byte(self):
        self.run_simple_test_shmoo(np.uint8)
    def test_2byte(self):
        self.run_simple_test_shmoo(np.uint16)
    def test_4byte(self):
        self.run_simple_test_shmoo(np.uint32)
    def test_8byte(self):
        self.run_simple_test_shmoo(np.uint64)
    def test_16byte(self):
        self.run_simple_test_shmoo(np.float128)
