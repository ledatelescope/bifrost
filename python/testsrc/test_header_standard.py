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
