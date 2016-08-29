"""@module test_late_resize
This file checks to see if resizing a ring after its sequence is opened
produces an error."""
import unittest
import json
import numpy as np
from bifrost.block import TestingBlock, SinkBlock, Pipeline

class ModWriteAsciiBlock(SinkBlock):
    """Copies input ring's data into ascii format
        in a text file."""
    def __init__(self, filename):
        """@param[in] filename Name of file to write ascii to"""
        self.filename = filename
        self.gulp_size = 1024
        open(self.filename, "w").close()
    def load_settings(self, input_header):
        header_dict = json.loads(input_header.tostring())
        self.shape = header_dict['shape']
        size_of_float32 = 4
        self.gulp_size = np.product(self.shape)*size_of_float32
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
        blocks.append((ModWriteAsciiBlock('.log.txt'), [0], []))
        Pipeline(blocks).main()
        np.testing.assert_almost_equal(
            np.loadtxt('.log.txt'), [1, 2, 3])
