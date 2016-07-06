# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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
from header_standard import enforce_header_standard
#from bifrost import header_standard

class TestHeaderStandardHandlesGoodHeaders(unittest.TestCase):
    def setUp(self):
        """Create empty header dictionary"""
        self.header_dict = {}
    def tearDown(self):
        """Make sure header is accepted"""
        self.assertTrue(
            enforce_header_standard(self.header_dict))
    def test_simple_header(self):
        self.header_dict = {
            'nchans':1, 'nifs':1, 'nbits':8, 'fch1':100.0, 'foff':1e-5,
            'tstart':1e5, 'tsamp':1e-5}
    def test_numpy_types(self):
        self.header_dict = {
            'nchans':1, 'nifs':1, 'nbits':8, 'fch1':100.0, 'foff':1e-5,
            'tstart':1e5, 'tsamp':1e-5}

class TestHeaderStandardHandlesBadHeaders(unittest.TestCase):
    def setUp(self):
        """Create empty header dictionary"""
        self.header_dict = {}
    def tearDown(self):
        """Make sure the header is rejected"""
        self.assertFalse(
            enforce_header_standard(self.header_dict))
    def test_empty_header(self):
        """Don't put anything in header"""
        pass
    def test_skip_one_parameter(self):
        """Make a good header, but without foff"""
        header_dict = {
            'nchans':1, 'nifs':1, 'nbits':8, 'fch1':100.0, 
            'tstart':1e5, 'tsamp':1e-5}
    def test_bad_nchans_types(self):
        """Put noninteger number of channels in header"""
        self.header_dict = {
            'nchans':1.05, 'nifs':1, 'nbits':8, 'fch1':100, 'foff':1e-5,
            'tstart':1e5, 'tsamp':1e-5}
    def test_bad_range(self):
        """Put in bad range of data"""
        self.header_dict = {
            'nchans':1.05, 'nifs':1, 'nbits':8, 'fch1':100, 'foff':1e-5,
            'tstart':1e5, 'tsamp':1e-5}

if __name__ == '__main__':
    unittest.main()
