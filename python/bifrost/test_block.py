from block import *
from bifrost.ring import Ring
import unittest

class TestCopyBlock(unittest.TestCase):
    """Performs tests of the Copy Block."""
    def setUp(self):
        self.input_ring = Ring()
        self.output_ring = Ring()
    def test_simple_ring_copy(self):
        logfile = '/data1/mcranmer/data/fake/1chan8bitNoDM.txt'
        blocks = []
        blocks.append(
            SigprocReadBlock(
                ['/data1/mcranmer/data/fake/1chan8bitNoDM.fil']))
        blocks.append(CopyBlock())
        blocks.append(WriteAsciiBlock(logfile))
        Pipeline(blocks).main()
        test_byte = open(logfile, 'r').read(1)
        self.assertEqual(test_byte, '2')
        

if __name__== "__main__":
    unittest.main()
