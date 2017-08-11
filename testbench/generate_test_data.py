"""
# generate_test_data.py

Generate test data that can be used with a testbench
"""
import os
import numpy as np

if __name__ == "__main__":

    if not os.path.exists('testdata'):
        os.mkdir('testdata')

    print "Generating sine wave dataset"
    for ii in range(32):
        # Generate test vector and save to file
        window_len = 2**16     
        n_window   = 32
        t = np.arange(window_len * n_window)
        w = 0.01
        s = np.sin((ii+1)* w * t, dtype='complex64')
        s.tofile('testdata/sin_data_%02i.bin' % ii)

    print "Generating noisy dataset"
    for ii in range(8):
        window_len = 2**18
        n_window   = 32
        noise = np.random.random(window_len * n_window).astype('complex64')
        t = np.arange(window_len * n_window, dtype='complex64')
        s0 = 1.0 * np.sin(0.01 * t, dtype='complex64')
        s1 = 2.0 * np.sin(0.07 * t, dtype='complex64')
        s2 = 4.0 * np.sin(0.12 * t, dtype='complex64')

        d = noise + s0 + s1 + s2
        d.tofile('testdata/noisy_data_%02i.bin' % ii)

