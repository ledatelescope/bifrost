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
from sigproc import *

class Test_init(unittest.TestCase):
    def setUp(self):
        self.myfile = SigprocFileRW()
        self.myfile.open(filename='/data1/mcranmer/data/fake/2chan1bitNoDM.fil',mode='r+b')
    def test_basic(self):
        self.myfile.close()
class Test_1bit(unittest.TestCase):
    def setUp(self):
        self.myfile = SigprocFileRW()
        self.myfile.open(filename='/data1/mcranmer/data/fake/2chan1bitNoDM.fil',mode='r+b')
    def tearDown(self):
        self.myfile.close()
    def test_read_write_data_equal(self):
        self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
        checkfile = SigprocFileRW()
        checkfile.open(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
        np.testing.assert_array_equal(self.myfile.data,checkfile.data)
        checkfile.close()
class Test_2bit(unittest.TestCase):
    def setUp(self):
        self.myfile = SigprocFileRW()
        self.myfile.open(filename='/data1/mcranmer/data/fake/2chan2bitNoDM.fil', mode='r+b')
    def tearDown(self):
        self.myfile.close()
    def test_read_write_data_equal(self):
        self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
        checkfile = SigprocFileRW()
        checkfile.open(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
        np.testing.assert_array_equal(self.myfile.data,checkfile.data)
        checkfile.close()
class Test_4bit(unittest.TestCase):
    def setUp(self):
        self.myfile = SigprocFileRW()
        self.myfile.open(filename='/data1/mcranmer/data/fake/2chan4bitNoDM.fil',mode='rb')
    def tearDown(self):
        self.myfile.close()
    def test_read_write_data_equal(self):
        self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
        checkfile = SigprocFileRW()
        checkfile.open(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
        np.testing.assert_array_equal(self.myfile.data,checkfile.data)
        checkfile.close()
class Test_8bit(unittest.TestCase):
    def setUp(self):
        self.myfile = SigprocFileRW()
        self.myfile.open(filename='/data1/mcranmer/data/fake/1chan8bitNoDM.fil',mode='r+b')
    def tearDown(self):
        self.myfile.close()
    def test_read_header(self):
        self.assertEqual(self.myfile.header,{'telescope_id': 4, 'refdm': 0.0, 'fch1': 433.968, 'data_type': 2, 'nchans': 1, 'tsamp': 8e-05, 'foff': -0.062, 'nbits': 8, 'header_size': 258, 'tstart': 50000.0, 'source_name': 'P: 3.141592700000 ms, DM: 0.000', 'nifs': 1, 'machine_id': 10})
    def test_read_frame_size(self):
        self.assertEqual(self.myfile.nifs,1)
    def test_data_test(self):
        data = self.myfile.data
        data = np.sum(data,axis=2)
        self.assertEqual(8495,data.T[0].sum(axis=0))
    def test_read_write_headers_equal(self):
        self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
        checkfile = SigprocFileRW()
        checkfile.open(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
        self.assertEqual(self.myfile.header, checkfile.header)
        checkfile.close()
    def test_read_write_data_equal(self):
        self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
        checkfile = SigprocFileRW()
        checkfile.open(filename='/data1/mcranmer/data/fake/test_write.fil', mode='rb')
        np.testing.assert_array_equal(self.myfile.data, checkfile.data)
        checkfile.close()
class Test_data_manip(unittest.TestCase):
    def setUp(self):
        self.my8bitfile = SigprocFileRW()
        self.my8bitfile.open(filename='/data1/mcranmer/data/fake/1chan8bitNoDM.fil',mode='r+b')
    def tearDown(self):
        self.my8bitfile.close()
    def test_append_data(self):
        initial_nframe = self.my8bitfile.nframe
        random_stream = np.random.randint(63, size=10000).astype('uint8').T
        self.my8bitfile.append_data(random_stream)
        self.assertEqual(self.my8bitfile.nframe,initial_nframe+10000)
    def test_append_untransposed_data(self):
        """test if appending data in different shape affects output"""
        initial_nframe = self.my8bitfile.nframe
        random_stream = np.random.randint(63, size=10000).astype('uint8')
        transposed_random_stream = random_stream.T
        transposeFile = self.my8bitfile
        transposeFile.append_data(transposed_random_stream)
        transposeFile.write_to('/data1/mcranmer/data/fake/test_file1.fil')
        self.my8bitfile.append_data(random_stream)
        self.my8bitfile.write_to('/data1/mcranmer/data/fake/test_file2.fil')
        self.assertEqual(SigprocFileRW().open('/data1/mcranmer/data/fake/test_file1.fil','rb').data.shape,\
                SigprocFileRW().open('/data1/mcranmer/data/fake/test_file1.fil','rb').data.shape)
class Test_16bit_2chan(unittest.TestCase):
    def setUp(self):
        self.my16bitfile = SigprocFileRW()
        self.my16bitfile.open(filename='/data1/mcranmer/data/fake/2chan16bitNoDM.fil',mode='r+b')
    def tearDown(self):
        self.my16bitfile.close()
    def test_data_read(self):
        """Test data to be 2 dimensions"""
        self.assertEqual(self.my16bitfile.data.shape[-1], 2)
    def test_append_2chan_data(self):
        initial_nframe = self.my16bitfile.nframe
        random_stream = np.random.randint(63, size=(10000,2)).astype('uint8')
        self.my16bitfile.append_data(random_stream)
        self.my16bitfile.write_to('/data1/mcranmer/data/fake/test_file1.fil')
        np.testing.assert_array_equal(self.my16bitfile.data,
                SigprocFileRW().open('/data1/mcranmer/data/fake/test_file1.fil','rb').data)
class Test_break_local_storage(unittest.TestCase):
    def setUp(self):
        self.myfile = SigprocFileRW()
        self.myfile.open('/data1/mcranmer/data/fake/256chan32bitNoDMLargeDuration.fil',mode='r+b')
    def tearDown(self):
        self.myfile.close()
    def test_data_read(self):
        """Test data to see if read in reasonable amount of time"""
        self.assertEqual(self.myfile.data.shape[-1], 2)
#Future tests:
# - make sigprocfile without a filename attached to it, and write data from it.
unittest.main()
