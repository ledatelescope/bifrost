"""
# test_fft.py

This testbench initializes a simple bifrost pipeline that reads from a binary file,
takes the FFT of the data (on the GPU no less), and then writes it to a new file. 
"""
import os
import glob
import numpy as np
import bifrost.pipeline as bfp
from bifrost.blocks import BinaryFileReadBlock, BinaryFileWriteBlock, FftBlock
from bifrost.blocks import CopyBlock, DetectBlock

from scipy.fftpack import fft as scipy_fft

if __name__ == "__main__":

    # FFT Parameters
    window_len = 2**18
    n_window   = 32

    # Setup pipeline
    filenames   = sorted(glob.glob('testdata/noisy_data*.bin'))

    b_read      = BinaryFileReadBlock(filenames, window_len, 1, 'cf32', core=0)
    b_copy      = CopyBlock(b_read, space='cuda', core=1, gpu=0)
    b_fft       = FftBlock(b_copy, axes=1, core=2, gpu=0)
    b_detect    = DetectBlock(b_fft, mode='scalar', axis=0, core=3)
    b_out       = CopyBlock(b_fft, space='system', core=4)
    b_write     = BinaryFileWriteBlock(b_out, core=5)

    # Run pipeline
    pipeline = bfp.get_default_pipeline()
    print pipeline.dot_graph()
    pipeline.run()

    # Check the output files match the input files
    for filename in filenames:
        try:
            print filename

            # Load the input data, do a windowed FFT
            indata  = np.fromfile(filename, dtype='complex64')
            indata  = scipy_fft(indata.reshape(n_window, window_len), axis=1)**2

            # Load the output data and reshape into windowed FFTs
            outdata = np.fromfile('%s.out' % filename, dtype='complex64')
            outdata = outdata.reshape(n_window, window_len)

            assert np.allclose(indata, outdata, atol=0.1)
            print "    Input data and output data match."
        except AssertionError:
            print "    Error: input and output data do not match."
            for ii in range(len(indata)):
                print "Window %02i match: %s" % (ii, np.allclose(indata[ii], outdata[ii], atol=0.1))
            print indata[0, 0:10]
            print outdata[0, 0:10]
            print np.max(indata - outdata)
        finally:
            print "    Cleaning up..."
            #os.remove(filename + '.out')
            print "    Done."