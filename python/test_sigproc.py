import unittest
from sigproc import *

class Test_1bit(unittest.TestCase):
	def setUp(self):
		self.myfile = SigprocFileRW(filename='/data1/mcranmer/data/fake/2chan1bitNoDM.fil')
		self.myfile.open(mode='r+b')
	def tearDown(self):
		self.myfile.close()
	def test_read_write_data_equal(self):
		self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
		checkfile = SigprocFileRW(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
		np.testing.assert_array_equal(self.myfile.data,checkfile.data)
		checkfile.close()
class Test_2bit(unittest.TestCase):
	def setUp(self):
		self.myfile = SigprocFileRW(filename='/data1/mcranmer/data/fake/2chan2bitNoDM.fil')
		self.myfile.open(mode='r+b')
	def tearDown(self):
		self.myfile.close()
	def test_read_write_data_equal(self):
		self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
		checkfile = SigprocFileRW(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
		np.testing.assert_array_equal(self.myfile.data,checkfile.data)
		checkfile.close()
class Test_4bit(unittest.TestCase):
	def setUp(self):
		self.myfile = SigprocFileRW(filename='/data1/mcranmer/data/fake/2chan4bitNoDM.fil')
		self.myfile.open(mode='r+b')
	
	def tearDown(self):
		self.myfile.close()

	def test_read_write_data_equal(self):
		self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
		checkfile = SigprocFileRW(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
		np.testing.assert_array_equal(self.myfile.data,checkfile.data)
		checkfile.close()

class Test_8bit(unittest.TestCase):
	def setUp(self):
		self.myfile = SigprocFileRW(filename='/data1/mcranmer/data/fake/1chan8bitNoDM.fil')
		self.myfile.open(mode='r+b')
	
	def tearDown(self):
		self.myfile.close()

	def test_read_header(self):
		self.assertEqual(self.myfile.header,{'telescope_id': 4, 'refdm': 0.0, 'fch1': 433.968, 'data_type': 2, 'nchans': 1, 'tsamp': 8e-05, 'foff': -0.062, 'nbits': 8, 'header_size': 258, 'tstart': 50000.0, 'source_name': 'P: 3.141592700000 ms, DM: 0.000', 'nifs': 1, 'machine_id': 10})

	def test_read_frame_size(self):
		self.assertEqual(self.myfile.frame_size,1)

	def test_data_test(self):
		data = self.myfile.data
		data = np.sum(data,axis=2)
		self.assertEqual(8495,data.T[0].sum(axis=0))
		
	def test_read_write_headers_equal(self):
		self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
		checkfile = SigprocFileRW(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
		self.assertEqual(self.myfile.header,checkfile.header)
		checkfile.close()
	
	def test_read_write_data_equal(self):
		self.myfile.write_to('/data1/mcranmer/data/fake/test_write.fil')
		checkfile = SigprocFileRW(filename='/data1/mcranmer/data/fake/test_write.fil',mode='rb')
		np.testing.assert_array_equal(self.myfile.data,checkfile.data)
		checkfile.close()





unittest.main()

