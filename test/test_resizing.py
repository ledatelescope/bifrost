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

"""@module test_resizing
This file checks different aspects of resizing a ring for segmentation faults."""
import unittest
import json
import numpy as np
from bifrost.block import TestingBlock, SinkBlock, Pipeline

class ModResizeAsciiBlock(SinkBlock):
    """Copies input ring's data into ascii format in a text file,
        after resizing late (after opening sequence)."""
    def __init__(self, filename, gulp_size=None):
        """@param[in] filename Name of file to write ascii to"""
        self.filename = filename
        self.gulp_size = gulp_size
        open(self.filename, "w").close()
    def load_settings(self, input_header):
        """Load the header, and set the gulp appropriately"""
        header_dict = json.loads(input_header.tostring())
        self.shape = header_dict['shape']
        size_of_float32 = 4
        if self.gulp_size is None:
            self.gulp_size = np.product(self.shape) * size_of_float32
    def iterate_ring_read(self, input_ring):
        """Iterate through one input ring
        @param[in] input_ring Ring to read through"""
        for sequence in input_ring.read(guarantee=True):
            self.load_settings(sequence.header)
            input_ring.resize(self.gulp_size)
            for span in sequence.read(self.gulp_size):
                yield span
    def main(self, input_ring):
        """Initiate the writing to file
        @param[in] input_rings First ring in this list will be used"""
        span_generator = self.iterate_ring_read(input_ring)
        span = span_generator.next()
        text_file = open(self.filename, 'a')
        np.savetxt(text_file, span.data_view(np.float32).reshape((1,-1)))

class TestLateResize(unittest.TestCase):
    """Test late resizing of a ring in a pipeline"""
    def test_modified_write_ascii(self):
        """Using a modified WriteAciiBlock, test the late resize.
        This should fail if ModWriteAscii block does not read the
        size of the input ring ahead of time, and resize accordingly."""
        blocks = []
        blocks.append((TestingBlock([1, 2, 3]), [], [0]))
        blocks.append((ModResizeAsciiBlock('.log.txt'), [0], []))
        Pipeline(blocks).main()
        np.testing.assert_almost_equal(
            np.loadtxt('.log.txt'), [1, 2, 3])

class TestLargeGulpSize(unittest.TestCase):
    """Create a gulp size larger than ring size"""
    def test_simple_large_gulp(self):
        """Test if a large gulp size produces a seg fault"""
        blocks = []
        blocks.append((TestingBlock([1, 2, 3]), [], [0]))
        blocks.append((ModResizeAsciiBlock('.log.txt', gulp_size=1024), [0], []))
        Pipeline(blocks).main()
        np.testing.assert_almost_equal(
            np.loadtxt('.log.txt'), [1, 2, 3])
