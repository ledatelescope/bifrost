"""This file wraps the bifrost FFT functions."""
from bifrost.libbifrost import _bf

def fft(input_data, output_data, direction="forward"):
    """Computes a fourier transform on input_data
    and returns it into output_data. Operates based
    on the dimensionality of the data itself.
    Assumes that input_data and output_data are both
    BFarrays"""
    if direction == "inverse":
        direction_code = 1
    else:
        direction_code = -1
    error = _bf.FFT(input_data, output_data, direction_code)
    return error()

def ifft(input_data, output_data):
    """Gives numpy syntax for bifrost, by simply
    passing the "inverse" string for the user
    into the fft function"""
    return fft(input_data, output_data, direction="inverse")
