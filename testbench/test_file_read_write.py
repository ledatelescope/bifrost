#!/usr/bin/env python3

# Copyright (c) 2017-2023, The Bifrost Authors. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of The Bifrost Authors nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
    print(pipeline.dot_graph())
    pipeline.run()

    # Check the output files match the input files
    for filename in filenames:
        try:
            print(filename)
            indata  = np.fromfile(filename, dtype='float32')
            outdata = np.fromfile('%s.out' % filename, dtype='float32')
            assert np.allclose(indata, outdata)
            print("    Input data and output data match.")
        except AssertionError:
            print("    Error: input and output data do not match.")
            raise
        finally:
            print("    Cleaning up...")
            os.remove(filename + '.out')
            print("    Done.")
