#This file wraps the bifrost FFT functions.
from bifrost.libbifrost import _bf
import ctypes

def fft(
    input_data, output_data, direction="forward"):
    """Computes a fourier transform on input_data
    and returns it into output_data. Operates based
    on the dimensionality of the data itself."""
    return _bf.FFT(
        input_data, output_data, 
        ctypes.cast(1, ctypes.c_int))

def ifft(input_data, output_data):
    """Gives numpy syntax for bifrost, by simply
    passing the "inverse" string for the user
    into the fft function"""
    return fft(input_data, output_data, "inverse")
