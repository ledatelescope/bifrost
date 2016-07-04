"""@package test_block
This file tests all aspects of the Bifrost.block module.
"""
import unittest
from block import *
from bifrost.ring import Ring

class TestCopyBlock(unittest.TestCase):
    """Performs tests of the Copy Block."""
    def setUp(self):
        self.input_ring = Ring()
        self.output_ring = Ring()
    def test_simple_ring_copy(self):
        """Test which performs a read of a sigproc file,
            copy to one ring, and then output as text."""
        logfile = 'log.txt'
        blocks = []
        blocks.append((
            SigprocReadBlock(
                ['/data1/mcranmer/data/fake/1chan8bitNoDM.fil']),
            [], [0]))
        blocks.append((CopyBlock(), [0], [1]))
        blocks.append((WriteAsciiBlock(logfile), [1], []))
        Pipeline(blocks).main()
        test_byte = open(logfile, 'r').read(1)
        self.assertEqual(test_byte, '2')
    def test_multi_linear_ring_copy(self):
        """Test which performs a read of a sigproc file,
            copy between many rings, and then output as
            text."""
        logfile = 'log2.txt'
        blocks = []
        blocks.append((
            SigprocReadBlock(
                ['/data1/mcranmer/data/fake/1chan8bitNoDM.fil']),
            [], [0]))
        blocks.append((CopyBlock(), [0], [1]))
        blocks.append((CopyBlock(), [1], [2]))
        blocks.append((CopyBlock(), [2], [3]))
        blocks.append((CopyBlock(), [3], [4]))
        blocks.append((WriteAsciiBlock(logfile), [4], []))
        Pipeline(blocks).main()
        test_byte = open(logfile, 'r').read(1)
        self.assertEqual(test_byte, '2')

if __name__ == "__main__":
    unittest.main()
