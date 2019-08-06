
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
import bifrost.unpack

class UnpackTest(unittest.TestCase):
    def run_unpack_to_ci8_test(self, iarray):
        oarray = bf.ndarray(shape=iarray.shape, dtype='ci8', space='cuda')
        oarray_known = bf.ndarray([[(0, 1), (2, 3)],
                                   [(4, 5), (6, 7)],
                                   [(-8, -7), (-6, -5)]],
                                  dtype='ci8')
        bf.unpack.unpack(iarray.copy(space='cuda'), oarray)
        oarray = oarray.copy(space='system')
        np.testing.assert_equal(oarray, oarray_known)
    def test_ci4_to_ci8(self):
        iarray = bf.ndarray([[(0x10,),(0x32,)],
                             [(0x54,),(0x76,)],
                             [(0x98,),(0xBA,)]],
                            dtype='ci4')
        self.run_unpack_to_ci8_test(iarray)
    def test_ci4_to_ci8_byteswap(self):
        iarray = bf.ndarray([[(0x01,),(0x23,)],
                             [(0x45,),(0x67,)],
                             [(0x89,),(0xAB,)]],
                            dtype='ci4')
        self.run_unpack_to_ci8_test(iarray.byteswap())
    def test_ci4_to_ci8_conjugate(self):
        iarray = bf.ndarray([[(0xF0,),(0xD2,)],
                             [(0xB4,),(0x96,)],
                             [(0x78,),(0x5A,)]],
                            dtype='ci4')
        self.run_unpack_to_ci8_test(iarray.conj())
    def test_ci4_to_ci8_byteswap_conjugate(self):
        iarray = bf.ndarray([[(0x0F,),(0x2D,)],
                             [(0x4B,),(0x69,)],
                             [(0x87,),(0xA5,)]],
                            dtype='ci4')
        self.run_unpack_to_ci8_test(iarray.byteswap().conj())
        
    def run_unpack_to_cf32_test(self, iarray):
        oarray = bf.ndarray(shape=iarray.shape, dtype='cf32', space='cuda')
        oarray_known = bf.ndarray([[ 0+1j,  2+3j],
                                   [ 4+5j,  6+7j],
                                   [-8-7j, -6-5j]],
                                  dtype='cf32')
        bf.unpack.unpack(iarray.copy(space='cuda'), oarray)
        oarray = oarray.copy(space='system')
        np.testing.assert_equal(oarray, oarray_known)
    def test_ci4_to_cf32(self):
        iarray = bf.ndarray([[(0x10,),(0x32,)],
                             [(0x54,),(0x76,)],
                             [(0x98,),(0xBA,)]],
                            dtype='ci4')
        self.run_unpack_to_cf32_test(iarray)
    def test_ci4_to_cf32_byteswap(self):
        iarray = bf.ndarray([[(0x01,),(0x23,)],
                             [(0x45,),(0x67,)],
                             [(0x89,),(0xAB,)]],
                            dtype='ci4')
        self.run_unpack_to_cf32_test(iarray.byteswap())
    def test_ci4_to_cf32_conjugate(self):
        iarray = bf.ndarray([[(0xF0,),(0xD2,)],
                             [(0xB4,),(0x96,)],
                             [(0x78,),(0x5A,)]],
                            dtype='ci4')
        self.run_unpack_to_cf32_test(iarray.conj())
    def test_ci4_to_cf32_byteswap_conjugate(self):
        iarray = bf.ndarray([[(0x0F,),(0x2D,)],
                             [(0x4B,),(0x69,)],
                             [(0x87,),(0xA5,)]],
                            dtype='ci4')
        self.run_unpack_to_cf32_test(iarray.byteswap().conj())