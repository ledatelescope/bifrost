"""
# test_file_read_write.py

This testbench initializes a simple bifrost pipeline that reads from a binary file,
and then writes the data to an output file. 
"""
import os
import numpy as np
import bifrost.pipeline as bfp
from bifrost.blocks import BinaryFileReadBlock, BinaryFileWriteBlock

if __name__ == "__main__":

    # Generate test vector and save to file
    t = np.arange(32768*1024)
    w = 0.01
    s0 = np.sin(w * t, dtype='float32')
    s0.tofile('numpy_data0.bin')
    s1 = np.sin(w * 4 * t, dtype='float32')
    s1.tofile('numpy_data1.bin')
    
    # Setup pipeline
    filenames   = ['numpy_data0.bin', 'numpy_data1.bin']
    b_read      = BinaryFileReadBlock(filenames, 32768, 1, 'f32')
    b_write     = BinaryFileWriteBlock(b_read.orings[0])

    # Run pipeline
    pipeline = bfp.get_default_pipeline()
    print pipeline.dot_graph()
    pipeline.run()
    
    # Check the output files match the input files
    outdata0 = np.fromfile('numpy_data0.bin.out', dtype='float32')
    outdata1 = np.fromfile('numpy_data1.bin.out', dtype='float32')
    
    try:
        assert np.allclose(s0, outdata0)
        assert np.allclose(s1, outdata1)
        print "Input data and output data match."
    except AssertionError:
        print "Error: input and output data do not match."
        raise
    finally:
        print "Cleaning up..."
        os.remove('numpy_data0.bin')
        os.remove('numpy_data1.bin')
        os.remove('numpy_data0.bin.out')
        os.remove('numpy_data1.bin.out')
        print "Done."