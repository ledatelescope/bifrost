
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
import bifrost.quantize

class QuantizeTest(unittest.TestCase):
    def run_quantize_from_cf32_test(self, out_dtype):
        iarray = bf.ndarray([[0.4 + 0.5j, 1.4 + 1.5j],
                             [2.4 + 2.5j, 3.4 + 3.5j],
                             [4.4 + 4.5j, 5.4 + 5.5j]],
                            dtype='cf32')
        oarray = bf.ndarray(shape=iarray.shape, dtype=out_dtype, space='cuda')
        oarray_known = bf.ndarray([[(0,0), (1,2)],
                                   [(2,2), (3,4)],
                                   [(4,4), (5,6)]],
                                  dtype=out_dtype)
        bf.quantize.quantize(iarray.copy(space='cuda'), oarray)
        oarray = oarray.copy(space='system')
        np.testing.assert_equal(oarray, oarray_known)
    def test_cf32_to_ci8(self):
        self.run_quantize_from_cf32_test('ci8')
    def test_cf32_to_ci16(self):
        self.run_quantize_from_cf32_test('ci16')
    def test_cf32_to_ci32(self):
        self.run_quantize_from_cf32_test('ci32')
