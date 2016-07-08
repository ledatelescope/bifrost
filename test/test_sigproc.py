import unittest
import bifrost
from bifrost.sigproc import *
import time

class Test_init(unittest.TestCase):
    def setUp(self):
        self.myfile = SigprocFile()
        self.myfile.open(filename='/data1/mcranmer/data/fake/2chan1bitNoDM.fil',mode='r+b')
        self.myfile.read_header()
    def test_basic(self):
        self.myfile.close()
class Test_1bit(unittest.TestCase):
    def setUp(self):
        self.myfile = SigprocFile()
        self.myfile.open(filename='/data1/mcranmer/data/fake/2chan1bitNoDM.fil',mode='r+b')
        self.myfile.read_header()
        self.myfile.read_data()
    def tearDown(self):
        self.myfile.close()
    def test_read_write_data_equal(self):
        self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
        checkfile = SigprocFile()
        checkfile.open(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
        checkfile.read_header()
        checkfile.read_data()
        np.testing.assert_array_equal(self.myfile.read_data(),checkfile.read_data())
        checkfile.close()
class Test_2bit(unittest.TestCase):
    def setUp(self):
        self.myfile = SigprocFile()
        self.myfile.open(filename='/data1/mcranmer/data/fake/2chan2bitNoDM.fil', mode='r+b')
        self.myfile.read_header()
        self.myfile.read_data()
    def tearDown(self):
        self.myfile.close()
    def test_read_write_data_equal(self):
        self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
        checkfile = SigprocFile()
        checkfile.open(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
        checkfile.read_header()
        checkfile.read_data()
        np.testing.assert_array_equal(self.myfile.read_data(),checkfile.read_data())
        checkfile.close()
class Test_4bit(unittest.TestCase):
    def setUp(self):
        self.myfile = SigprocFile()
        self.myfile.open(filename='/data1/mcranmer/data/fake/2chan4bitNoDM.fil',mode='rb')
        self.myfile.read_header()
        self.myfile.read_data()
    def tearDown(self):
        self.myfile.close()
    def test_read_write_data_equal(self):
        self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
        checkfile = SigprocFile()
        checkfile.open(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
        checkfile.read_header()
        checkfile.read_data()
        np.testing.assert_array_equal(self.myfile.read_data(),checkfile.read_data())
        checkfile.close()
class Test_8bit(unittest.TestCase):
    def setUp(self):
        self.myfile = SigprocFile()
        self.myfile.open(filename='/data1/mcranmer/data/fake/1chan8bitNoDM.fil',mode='r+b')
        self.myfile.read_header()
        self.myfile.read_data()
    def tearDown(self):
        self.myfile.close()
    def test_read_header(self):
        self.assertEqual(self.myfile.header,{'telescope_id': 4, 'refdm': 0.0, 'fch1': 433.968, 'data_type': 2, 'nchans': 1, 'tsamp': 8e-05, 'foff': -0.062, 'nbits': 8, 'header_size': 258, 'tstart': 50000.0, 'source_name': 'P: 3.141592700000 ms, DM: 0.000', 'nifs': 1, 'machine_id': 10})
    def test_read_frame_size(self):
        self.assertEqual(self.myfile.nifs,1)
    def test_data_test(self):
        data = self.myfile.read_data()
        data = np.sum(data,axis=2)
        self.assertEqual(70552,data.T[0].sum(axis=0))
    def test_read_write_headers_equal(self):
        self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
        checkfile = SigprocFile()
        checkfile.open(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
        checkfile.read_header()
        checkfile.read_data()
        self.assertEqual(self.myfile.header, checkfile.header)
        checkfile.close()
    def test_read_write_data_equal(self):
        self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
        checkfile = SigprocFile()
        checkfile.open(filename='/data1/mcranmer/data/fake/test_write.fil', mode='rb')
        checkfile.read_header()
        checkfile.read_data()
        np.testing.assert_array_equal(self.myfile.read_data(), checkfile.read_data())
        checkfile.close()
class Test_data_manip(unittest.TestCase):
    def setUp(self):
        self.my8bitfile = SigprocFile()
        self.my8bitfile.open(filename='/data1/mcranmer/data/fake/1chan8bitNoDM.fil',mode='r+b')
        self.my8bitfile.read_header()
        self.my8bitfile.read_data()
    def tearDown(self):
        self.my8bitfile.close()
    def test_append_data(self):
        testfile = SigprocFile()
        testfile.open(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
        testfile.data = self.my8bitfile.read_data()
        testfile.header = self.my8bitfile.header
        testfile.interpret_header()
        self.my8bitfile.clear()
        testfile.write_to('/data1/mcranmer/data/fake/1chan8bitNoDM.fil')
        initial_nframe = testfile.get_nframe()
        random_stream = np.random.randint(63, size=10000).astype('uint8').T
        testfile.append_data(random_stream)
        self.assertEqual(testfile.data.shape[0],initial_nframe+10000)
    def test_data_slice(self):
        testFile = SigprocFile()
        testFile.open(filename='/data1/mcranmer/data/fake/1chan8bitNoDM.fil', mode='r+b')
        testFile.read_header()
        self.assertEqual(testFile.read_data(-1).shape, (1, 1, 1))
    def test_append_untransposed_data(self):
        """test if appending data in different shape affects output"""
        initial_nframe = self.my8bitfile.get_nframe()
        random_stream = np.random.randint(63, size=10000).astype('uint8')
        transposed_random_stream = random_stream.T
        testfile1 = SigprocFile()
        testfile1.data = self.my8bitfile.read_data()
        testfile1.header = self.my8bitfile.header
        testfile1.interpret_header()
        testfile2 = SigprocFile()
        testfile2.data = self.my8bitfile.read_data()
        testfile2.header = self.my8bitfile.header
        testfile2.interpret_header()
        testfile1.append_data(random_stream)
        testfile2.append_data(transposed_random_stream)
        testfile1.write_to(filename='/data1/mcranmer/data/fake/test_write1.fil')
        testfile2.write_to(filename='/data1/mcranmer/data/fake/test_write2.fil')
        testfile1.open(filename='/data1/mcranmer/data/fake/test_write1.fil', mode='rb')
        testfile2.open(filename='/data1/mcranmer/data/fake/test_write1.fil', mode='rb')
        np.testing.assert_array_equal(testfile1.read_data(), testfile2.read_data())
class Test_16bit_2chan(unittest.TestCase):
    def setUp(self):
        self.my16bitfile = SigprocFile()
        self.my16bitfile.open(filename='/data1/mcranmer/data/fake/2chan16bitNoDM.fil',mode='r+b')
        self.my16bitfile.read_header()
        self.my16bitfile.read_data()
    def tearDown(self):
        self.my16bitfile.close()
    def test_data_read(self):
        """Test data to be 2 dimensions"""
        self.assertEqual(self.my16bitfile.read_data().shape[-1], 2)
    def test_append_2chan_data(self):
        initial_nframe = self.my16bitfile.get_nframe()
        random_stream = np.random.randint(2**16-1, size=(10000,2)).astype('uint16')
        file1 = SigprocFile()
        file1.header = self.my16bitfile.header
        file1.interpret_header()
        file1.data = self.my16bitfile.data
        file1.write_to('/data1/mcranmer/data/fake/test_file1.fil')
        file1.open('/data1/mcranmer/data/fake/test_file1.fil','r+b')
        file1.append_data(random_stream)
        file1.write_to('/data1/mcranmer/data/fake/test_file2.fil')
        file2 = SigprocFile().open('/data1/mcranmer/data/fake/test_file2.fil','rb')
        file2.read_header()
        file2.read_data()
        np.testing.assert_array_equal(file1.read_data(),file2.read_data())
class Test_break_local_storage(unittest.TestCase):
    def setUp(self):
        self.myfile = SigprocFile()
        self.myfile.open('/data1/mcranmer/data/fake/256chan32bitNoDMLargeDuration.fil',mode='r+b')
        self.myfile.read_header()
    def tearDown(self):
        self.myfile.close()
    def test_data_read_timed(self):
        """Test data to see if read in reasonable amount of time"""
        start_time = time.clock()
        data = self.myfile.read_data(-3)
        assert time.clock()-start_time < 0.1, "Taking too long to read. (Probably reading in entire file.)"
        self.assertEqual(data.shape[0], 3)
    def test_data_read_timed_2(self):
        start_time = time.clock()
        data = self.myfile.read_data(0, 3)
        assert time.clock()-start_time < 0.1, "Taking too long to read. (Probably reading in entire file.)"
        self.assertEqual(data.shape[0], 3)
    def test_data_read_timed_3(self):
        start_time = time.clock()
        data = self.myfile.read_data(-30000, -30000+3)
        assert time.clock()-start_time < 0.1, "Taking too long to read. (Probably reading in entire file.)"
        self.assertEqual(data.shape[0], 3)
    def test_data_read_timed_4(self):
        start_time = time.clock()
        data = self.myfile.read_data(30000, 30000+3)
        assert time.clock()-start_time < 0.1, "Taking too long to read. (Probably reading in entire file.)"
        self.assertEqual(data.shape[0], 3)
class Test_data_slicing(unittest.TestCase):
    def setUp(self):
        self.myfile = SigprocFile()
        self.myfile.open('/data1/mcranmer/data/fake/1chan8bitNoDM.fil',mode='r+b')
        self.myfile.read_header()
    def tearDown(self):
        self.myfile.close()
    def test_only_negative_end_given(self):
        data = self.myfile.read_data(end=-3)
        self.assertEqual(data.shape[1:],(1,1))
        self.assertTrue(data.shape[0] > 100) #assumes more than 100 frames in .fil
    def test_different_signs(self):
        print self.myfile.read_data().shape
        data = self.myfile.read_data(3,-3)
        self.assertEqual(data.shape,(12800-6, 1, 1)) #assumes more than ~100 frames in .fil

#Future tests:
# - make sigprocfile without a filename attached to it, and write data from it.
unittest.main()
