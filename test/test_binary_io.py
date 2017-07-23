# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
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

import unittest
import bifrost as bf
import numpy as np
from bifrost import blocks
from bifrost import pipeline as bfp

class CallbackBlock(blocks.CopyBlock):
    """Testing-only block which calls user-defined
        functions on sequence and on data"""
    def __init__(self, iring, seq_callback, data_callback, data_ref=None,
                 *args, **kwargs):
        super(CallbackBlock, self).__init__(iring, *args, **kwargs)
        self.seq_callback  = seq_callback
        self.data_callback = data_callback
        self.data_ref = data_ref
    def on_sequence(self, iseq):
        self.seq_callback(iseq)
        return super(CallbackBlock, self).on_sequence(iseq)
    def on_data(self, ispan, ospan):
        self.data_callback(ispan, ospan)
        if self.data_ref is not None:
            # Note: This can be used to check data from outside the pipeline,
            #         which is useful when exceptions inside blocks prevent
            #         downstream callback blocks from ever executing.
            self.data_ref['odata'] = ospan.data.copy()
        return super(CallbackBlock, self).on_data(ispan, ospan)

class BinaryIOTest(unittest.TestCase):
    """Test simple IO for the binary read/write blocks"""
    def setUp(self):
        """Generate some dummy data to read"""
        # Generate test vector and save to file
        t = np.arange(32768 * 1024)
        w = 0.01
        self.s0 = np.sin(w * t, dtype='float32')
        self.s0.tofile('numpy_data0.bin')
        self.s1 = np.sin(w * 4 * t, dtype='float32')
        self.s1.tofile('numpy_data1.bin')

        # Setup pipeline
        self.filenames = ['numpy_data0.bin', 'numpy_data1.bin']
    def test_read_write(self):
        """Read from a binary file, then write to another one"""
        b_read = blocks.binary_read(self.filenames, 32768, 1, 'f32')
        b_write = blocks.binary_write(b_read.orings[0])

        # Run pipeline
        pipeline = bfp.get_default_pipeline()
        pipeline.run()

        # Check the output files match the input files
        outdata0 = np.fromfile('numpy_data0.bin.out', dtype='float32')
        outdata1 = np.fromfile('numpy_data1.bin.out', dtype='float32')

        np.testing.assert_almost_equal(self.s0, outdata0)
        np.testing.assert_almost_equal(self.s1, outdata1)
    def test_header_compliance(self):
        """Make sure that the binary in has all required header labels"""
        def seq_callback(iseq):
            for key in ['_tensor']:
                self.assertTrue(bool(key in iseq.header))
            for key in ['units', 'labels', 'scales', 'dtype', 'shape']:
                self.assertTrue(key in iseq.header['_tensor'])
            self.assertTrue(type(iseq.header['_tensor']['scales'][0]) is list)
        def data_callback(ispan, ospan):
            pass

        self.fil_file = "./data/2chan16bitNoDM.fil"
        with bf.Pipeline() as pipeline:
            b_read = blocks.binary_read(self.filenames, 32768, 1, 'f32')
            callback = CallbackBlock(b_read, seq_callback, data_callback)

            pipeline.run()
