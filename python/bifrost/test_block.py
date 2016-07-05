"""@package test_block
This file tests all aspects of the Bifrost.block module.
"""
import unittest
from block import *
from bifrost.ring import Ring

class TestCopyBlock(unittest.TestCase):
    """Performs tests of the Copy Block."""
    def setUp(self):
        """Set up the blocks list, and put in a single
            block which reads in the data from a filterbank 
            file."""
        self.blocks = []
        self.blocks.append((
            SigprocReadBlock(
                ['/data1/mcranmer/data/fake/1chan8bitNoDM.fil']),
            [], [0]))
    def SimpleSigprocCopyAndDump_DumpsFirstByteAs2(self):
        """Test which performs a read of a sigproc file,
            copy to one ring, and then output as text."""
        logfile = 'log.txt'
        self.blocks.append((CopyBlock(), [0], [1]))
        self.blocks.append((WriteAsciiBlock(logfile), [1], []))
        Pipeline(self.blocks).main()
        test_byte = open(logfile, 'r').read(1)
        self.assertEqual(test_byte, '2')
    def CopyBetweenMultipleBlocksLinear_DumpFirstByte2(self):
        """Test which performs a read of a sigproc file,
            copy between many rings, and then output as
            text."""
        logfile = 'log2.txt'
        for i in range(10):
            self.blocks.append(
                (CopyBlock(), [i], [i+1]))
        self.blocks.append((WriteAsciiBlock(logfile), [10], []))
        Pipeline(self.blocks).main()
        test_byte = open(logfile, 'r').read(1)
        self.assertEqual(test_byte, '2')
    def CopyBetweenMultipleBlocksNonlinear_Dump17thByte3(self):
        """Test which reads in a sigproc file, and
            loads it between different rings in a 
            nonlinear fashion, then outputs to file."""
        logfile = 'log3.txt'
        self.blocks.append((CopyBlock(), [0], [1]))
        self.blocks.append((CopyBlock(), [0], [2]))
        self.blocks.append((CopyBlock(), [2], [5]))
        self.blocks.append((CopyBlock(), [0], [3]))
        self.blocks.append((CopyBlock(), [3], [4]))
        self.blocks.append((CopyBlock(), [5], [6]))
        self.blocks.append((WriteAsciiBlock(logfile), [6], []))
        Pipeline(self.blocks).main()
        test_byte = open(logfile, 'r').read(17)
        self.assertEqual(test_byte[-1:], '3')

if __name__ == "__main__":
    unittest.main()
