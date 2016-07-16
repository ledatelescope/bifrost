import unittest
from blocks import DadaReadBlock
import bifrost
from bifrost.block import *

class TestDadaBlock(unittest.TestCase):
    """Test the ability of the Dada block to read
        in data that is compatible with other blocks."""
    def setUp(self):
        self.blocks = []
        self.blocks.append(
            (DadaReadBlock(
                "/data1/mcranmer/data/real/leda/2016_xaa.dada"),
            [], [0]))
    def test_read_and_write(self):
        """Reads in a dada file, and logs in ascii
            file."""
        logfile = '.log.txt'
        self.blocks.append((WriteAsciiBlock(logfile), [0], []))
        Pipeline(self.blocks).main() 
        test_bytes = open(logfile, 'r').read(500).split(' ')
        self.assertAlmostEqual(np.float(test_bytes[0]), 3908.5, 3)
    def test_read_copy_write(self):
        """Adds another intermediate block to the
            last step."""
        logfile = '.log.txt'
        self.blocks.append((CopyBlock(), [0], [1, 2, 3]))
        self.blocks.append((WriteAsciiBlock(logfile), [3], []))
        Pipeline(self.blocks).main() 
        test_bytes = open(logfile, 'r').read(500).split(' ')
        self.assertAlmostEqual(np.float(test_bytes[0]), 3908.5, 3)

if __name__ == "__main__":
    unittest.main()
