"""
# test_file_read_write.py

This testbench initializes a simple bifrost pipeline that reads from a binary file,
and then writes the data to an output file. 
"""
import os
import numpy as np
import bifrost.pipeline as bfp
from bifrost.blocks import BinaryFileReadBlock, BinaryFileWriteBlock
import glob


if __name__ == "__main__":

    # Setup pipeline
    filenames   = sorted(glob.glob('testdata/sin_data*.bin'))
    b_read      = BinaryFileReadBlock(filenames, 32768, 1, 'f32')
    b_write     = BinaryFileWriteBlock(b_read)

    # Run pipeline
    pipeline = bfp.get_default_pipeline()
    print pipeline.dot_graph()
    pipeline.run()

    # Check the output files match the input files
    for filename in filenames:
        try:
            print filename
            indata  = np.fromfile(filename, dtype='float32')
            outdata = np.fromfile('%s.out' % filename, dtype='float32')
            assert np.allclose(indata, outdata)
            print "    Input data and output data match."
        except AssertionError:
            print "    Error: input and output data do not match."
            raise
        finally:
            print "    Cleaning up..."
            os.remove(filename + '.out')
            print "    Done."
