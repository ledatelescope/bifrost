"""@package test_header_standard
This file tests the header standard"""
import unittest
import numpy as np
from bifrost.header_standard import enforce_header_standard

class TestHeaderStandardHandlesGoodHeaders(unittest.TestCase):
    def setUp(self):
        """Create empty header dictionary"""
        self.header_dict = {}
    def tearDown(self):
        """Make sure header is accepted"""
        self.assertTrue(
            enforce_header_standard(self.header_dict))
    def test_simple_header(self):
        """Simple header test, with all good values in range"""
        self.header_dict = {
            'nchans':1, 'nifs':1, 'nbits':8, 'fch1':100.0, 'foff':1e-5,
            'tstart':1e5, 'tsamp':1e-5}
    def test_numpy_types(self):
        """Same values, but some are numpy types"""
        self.header_dict = {
            'nchans':np.int(1), 'nifs':1, 'nbits':8, 
            'fch1':np.float(100.0), 'foff':np.float(1e-5),
            'tstart':1e5, 'tsamp':np.float(1e-5)}
    def test_extra_parameters(self):
        """Add some extra parameters"""
        self.header_dict = {
            'nchans':1, 'nifs':1, 'nbits':8, 'fch1':100.0, 'foff':1e-5,
            'tstart':1e5, 'tsamp':1e-5, 'my_extra_param':50}

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
    def test_low_value(self):
        """Put in low value for nbits"""
        self.header_dict = {
            'nchans':1, 'nifs':1, 'nbits':-8, 'fch1':100.0, 'foff':1e-5,
            'tstart':1e5, 'tsamp':1e-5}

if __name__ == '__main__':
    unittest.main()
