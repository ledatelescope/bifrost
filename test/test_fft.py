"""This set of unit tests check the functionality
on the bifrost FFT wrapper, both in device memory and out"""
import ctypes
import unittest
import numpy as np
from bifrost.GPUArray import GPUArray
from bifrost.ring import Ring
from bifrost.fft import fft, ifft
from bifrost.libbifrost import _bf, _string2space

class TestFFTHandles1DComplex(unittest.TestCase):
    """This test runs one dimensional complex data
    in device memory through an FFT, forward and then
    inverse."""
    def setUp(self):
        """Create two arrays in device memory, input_data with
        defined data"""
        self.input_data = GPUArray(shape=5, dtype=np.complex64)
        self.output_data = GPUArray(shape=5, dtype=np.complex64)
        defined_data = np.array([0, 0, 10, 0, -5j]).astype(np.complex64)
        self.input_data.set(defined_data)
        self.output_data.set(1j*np.arange(5).astype(np.complex64))
    def test_forwardfft(self):
        """Computes a forward FFT and checks accuracy of result"""
        input_array = self.input_data.as_BFarray(3)
        output_array = self.output_data.as_BFarray(3)
        fft(input_array, output_array)
        self.output_data.buffer = output_array.data
        local_data = self.output_data.get()
        self.assertAlmostEqual(local_data[0],10-5j,places=4)
    def test_inversefft(self):
        """Computes an inverse FFT and checks accuracy of result"""
        input_array = self.input_data.as_BFarray(3)
        output_array = self.output_data.as_BFarray(3)
        ifft(input_array, output_array)
        self.output_data.buffer = output_array.data
        local_data = self.output_data.get()
        self.assertAlmostEqual(local_data[1],-12.8455+4.33277j,places=4)




