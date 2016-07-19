import unittest
from block import *

class TestFFTBlock(unittest.TestCase):
    """This test assures basic functionality of fft block"""
    def setUp(self):
        """Assemble a basic pipeline with the FFT block"""
        self.logfile = '.log.txt'
        self.blocks = []
        self.blocks.append((
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/1chan8bitNoDM.fil'),
            [], [0]))
        self.blocks.append((FFTBlock(gulp_size=4096*8*8*8), [0], [1]))
        self.blocks.append((WriteAsciiBlock(self.logfile), [1], []))
    def test_throughput(self):
        """Test that any data is being put through"""
        Pipeline(self.blocks).main()
        test_string = open(self.logfile, 'r').read()
        self.assertGreater(len(test_string), 0)
    def test_throughput_size(self):
        """Number of elements going out should be double that of basic copy"""
        Pipeline(self.blocks).main()
        number_fftd = len(open(self.logfile, 'r').read().split(' '))
        open(self.logfile, 'w').close()
        ## Run pipeline again with simple copy
        self.blocks[1] = (CopyBlock(), [0], [1])
        Pipeline(self.blocks).main()
        #number_copied = len(open(self.logfile, 'r').read().split(' '))
        #self.assertEqual(number_fftd, 2*number_copied)
    def test_data_sizes(self):
        """Test that different number of bits give correct throughput size"""
        for iterate in range(5):
            nbit = 2**iterate
            if nbit == 8:
                continue
            self.blocks[0] = (
                SigprocReadBlock(
                    '/data1/mcranmer/data/fake/2chan'+ str(nbit) + 'bitNoDM.fil'),
                [], [0])
            open(self.logfile, 'w').close()
            Pipeline(self.blocks).main()
            number_fftd = np.loadtxt(self.logfile).astype(np.float32).view(np.complex64).size
            # Compare with simple copy
            self.blocks[1] = (CopyBlock(), [0], [1])
            open(self.logfile, 'w').close()
            Pipeline(self.blocks).main()
            number_copied = np.loadtxt(self.logfile).size
            self.assertEqual(number_fftd, number_copied)
            # Go back to FFT
            self.blocks[1] = (FFTBlock(gulp_size=4096*8*8*8), [0], [1])
    def test_fft_result(self):
        """Make sure that fft matches what it should!"""
        open(self.logfile, 'w').close()
        Pipeline(self.blocks).main()
        fft_block_result = np.loadtxt(self.logfile).astype(np.float32).view(np.complex64)
        self.blocks[1] = (CopyBlock(), [0], [1])
        open(self.logfile, 'w').close()
        Pipeline(self.blocks).main()
        normal_fft_result = np.fft.fft(np.loadtxt(self.logfile))
        np.testing.assert_almost_equal(fft_block_result, normal_fft_result, 2)
class TestIFFTBlock(unittest.TestCase):
    """This test assures basic functionality of the ifft block.
    Requires the FFT block for testing."""
    def setUp(self):
        """Assemble a basic pipeline with the FFT/IFFT blocks"""
        self.logfile = '.log.txt'
        self.blocks = []
        self.blocks.append((
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/1chan8bitNoDM.fil'),
            [], [0]))
        self.blocks.append((FFTBlock(gulp_size=4096*8*8*8*8), [0], [1]))
        self.blocks.append((IFFTBlock(gulp_size=4096*8*8*8*8), [1], [2]))
        self.blocks.append((WriteAsciiBlock(self.logfile), [2], []))
    def test_equivalent_data_to_copy(self):
        """Test that the data coming out of this pipeline is equivalent
        the initial read data"""
        open(self.logfile, 'w').close()
        Pipeline(self.blocks).main()
        unfft_result = np.loadtxt(self.logfile).astype(np.float32).view(np.complex64)
        self.blocks[1] = (CopyBlock(), [0], [1])
        self.blocks[2] = (WriteAsciiBlock(self.logfile), [1], [])
        del self.blocks[3]
        open(self.logfile, 'w').close()
        Pipeline(self.blocks).main()
        untouched_result = np.loadtxt(self.logfile).astype(np.float32)
        np.testing.assert_almost_equal(unfft_result, untouched_result, 2)
