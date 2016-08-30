"""@package test_block
This file tests all aspects of the Bifrost.block module.
"""
import unittest
import numpy as np
from bifrost.ring import Ring
from bifrost.block import TestingBlock, WriteAsciiBlock, WriteHeaderBlock
from bifrost.block import SigprocReadBlock, CopyBlock, KurtosisBlock, FoldBlock
from bifrost.block import IFFTBlock, FFTBlock, Pipeline, MultiAddBlock
from bifrost.block import SplitterBlock, NumpyBlock, NumpySourceBlock

class TestIterateRingWrite(unittest.TestCase):
    """Test the iterate_ring_write function of SourceBlocks/TransformBlocks"""
    def test_throughput(self):
        """Read in data with a small throughput size. Expect all to go through."""
        blocks = []
        blocks.append((
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/1chan8bitNoDM.fil', gulp_nframe=4096),
            [], [0]))
        blocks.append((WriteAsciiBlock('.log.txt'), [0], []))
        Pipeline(blocks).main()
        log_data = np.loadtxt('.log.txt')
        self.assertEqual(log_data.size, 12800)
class TestTestingBlock(unittest.TestCase):
    """Test the TestingBlock for basic functionality"""
    def setUp(self):
        """Initiate blocks list with write asciiBlock"""
        self.blocks = []
        self.blocks.append((WriteAsciiBlock('.log.txt', gulp_size=3*4), [0], []))
    def test_simple_dump(self):
        """Input some numbers, and ensure they are written to a file"""
        self.blocks.append((TestingBlock([1, 2, 3]), [], [0]))
        Pipeline(self.blocks).main()
        dumped_numbers = np.loadtxt('.log.txt')
        np.testing.assert_almost_equal(dumped_numbers, [1, 2, 3])
    def test_multi_dimensional_input(self):
        """Input a 2 dimensional list, and have this printed"""
        test_array = [[1, 2], [3, 4]]
        self.blocks[0] = (WriteAsciiBlock('.log.txt', gulp_size=4*4), [0], [])
        self.blocks.append((TestingBlock(test_array), [], [0]))
        self.blocks.append((WriteHeaderBlock('.log2.txt'), [0], []))
        Pipeline(self.blocks).main()
        header = eval(open('.log2.txt').read()) #pylint:disable=eval-used
        dumped_numbers = np.loadtxt('.log.txt').reshape(header['shape'])
        np.testing.assert_almost_equal(dumped_numbers, test_array)
class TestCopyBlock(unittest.TestCase):
    """Performs tests of the Copy Block."""
    def setUp(self):
        """Set up the blocks list, and put in a single
            block which reads in the data from a filterbank
            file."""
        self.blocks = []
        self.blocks.append((
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/1chan8bitNoDM.fil'),
            [], [0]))
    def test_simple_copy(self):
        """Test which performs a read of a sigproc file,
            copy to one ring, and then output as text."""
        logfile = '.log.txt'
        self.blocks.append((CopyBlock(), [0], [1]))
        self.blocks.append((WriteAsciiBlock(logfile), [1], []))
        Pipeline(self.blocks).main()
        test_byte = open(logfile, 'r').read(1)
        self.assertEqual(test_byte, '2')
    def test_multi_copy(self):
        """Test which performs a read of a sigproc file,
            copy between many rings, and then output as
            text."""
        logfile = '.log.txt'
        for i in range(10):
            self.blocks.append(
                (CopyBlock(), [i], [i+1]))
        self.blocks.append((WriteAsciiBlock(logfile), [10], []))
        Pipeline(self.blocks).main()
        test_byte = open(logfile, 'r').read(1)
        self.assertEqual(test_byte, '2')
    def test_non_linear_multi_copy(self):
        """Test which reads in a sigproc file, and
            loads it between different rings in a
            nonlinear fashion, then outputs to file."""
        logfile = '.log.txt'
        self.blocks.append((CopyBlock(), [0], [1]))
        self.blocks.append((CopyBlock(), [0], [2]))
        self.blocks.append((CopyBlock(), [2], [5]))
        self.blocks.append((CopyBlock(), [0], [3]))
        self.blocks.append((CopyBlock(), [3], [4]))
        self.blocks.append((CopyBlock(), [5], [6]))
        self.blocks.append((WriteAsciiBlock(logfile), [6], []))
        Pipeline(self.blocks).main()
        log_nums = open(logfile, 'r').read(500).split(' ')
        test_num = np.float(log_nums[8])
        self.assertEqual(test_num, 3)
    def test_single_block_multi_copy(self):
        """Test which forces one block to do multiple
            copies at once, and then dumps to two files,
            checking them both."""
        logfiles = ['.log1.txt', '.log2.txt']
        self.blocks.append((CopyBlock(), [0], [1, 2]))
        self.blocks.append((WriteAsciiBlock(logfiles[0]), [1], []))
        self.blocks.append((WriteAsciiBlock(logfiles[1]), [2], []))
        Pipeline(self.blocks).main()
        test_bytes = int(
            open(logfiles[0], 'r').read(1)) + int(
                open(logfiles[1], 'r').read(1))
        self.assertEqual(test_bytes, 4)
    def test_32bit_copy(self):
        """Perform a simple test to confirm that 32 bit
            copying has no information loss"""
        logfile = '.log.txt'
        self.blocks = []
        self.blocks.append((
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/256chan32bitNoDM.fil'),
            [], [0]))
        self.blocks.append((CopyBlock(), [0], [1]))
        self.blocks.append((WriteAsciiBlock(logfile), [1], []))
        Pipeline(self.blocks).main()
        test_bytes = open(logfile, 'r').read(500).split(' ')
        self.assertAlmostEqual(np.float(test_bytes[0]), 0.72650784254)
class TestFoldBlock(unittest.TestCase):
    """This tests functionality of the FoldBlock."""
    def setUp(self):
        """Set up the blocks list, and put in a single
            block which reads in the data from a filterbank
            file."""
        self.blocks = []
        self.blocks.append((
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/pulsar_noisey_NoDM.fil'),
            [], [0]))
    def dump_ring_and_read(self):
        """Dump block to ring, read in as histogram"""
        logfile = ".log.txt"
        self.blocks.append((WriteAsciiBlock(logfile), [1], []))
        Pipeline(self.blocks).main()
        test_bytes = open(logfile, 'r').read().split(' ')
        histogram = np.array([np.float(x) for x in test_bytes])
        return histogram
    def test_simple_pulsar(self):
        """Test whether a pulsar histogram
            shows a large peak and is mostly
            nonzero values"""
        self.blocks.append((
            FoldBlock(bins=100), [0], [1]))
        histogram = self.dump_ring_and_read()
        self.assertEqual(histogram.size, 100)
        self.assertTrue(np.min(histogram) > 1e-10)
    def test_different_bin_size(self):
        """Try a different bin size"""
        self.blocks.append((
            FoldBlock(bins=50), [0], [1]))
        histogram = self.dump_ring_and_read()
        self.assertEqual(histogram.size, 50)
    def test_show_pulse(self):
        """Test to see if a pulse is visible in the
            histogram from pulsar data"""
        self.blocks[0] = (
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/simple_pulsar_DM0.fil'),
            [], [0])
        self.blocks.append((
            FoldBlock(bins=200), [0], [1]))
        histogram = self.dump_ring_and_read()
        self.assertTrue(np.min(histogram) > 1e-10)
        self.assertGreater(
            np.max(histogram)/np.average(histogram), 5)
    def test_many_channels(self):
        """See if many channels work with folding"""
        self.blocks[0] = (
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/simple_pulsar_DM0_128ch.fil'),
            [], [0])
        self.blocks.append((
            FoldBlock(bins=200), [0], [1]))
        histogram = self.dump_ring_and_read()
        self.assertTrue(np.min(histogram) > 1e-10)
        self.assertGreater(
            np.max(histogram)/np.min(histogram), 3)
    def test_high_dispersion(self):
        """Test folding on a file with high DM"""
        self.blocks[0] = (
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/simple_pulsar_DM10_128ch.fil'),
            [], [0])
        self.blocks.append((
            FoldBlock(bins=200, dispersion_measure=10, core=0),
            [0], [1]))
        histogram = self.dump_ring_and_read()
        self.assertTrue(np.min(histogram) > 1e-10)
        self.assertGreater(
            np.max(histogram)/np.min(histogram), 3)
        #TODO: Test to break bfmemcpy2D for lack of float32 functionality?
class TestKurtosisBlock(unittest.TestCase):
    """This tests functionality of the KurtosisBlock."""
    def test_data_throughput(self):
        """Check that data is being put through the block
        (does this by checking consistency of shape/datatype)"""
        blocks = []
        blocks.append((
            SigprocReadBlock('/data1/mcranmer/data/fake/1chan8bitNoDM.fil'),
            [], [0]))
        blocks.append((
            KurtosisBlock(), [0], [1]))
        blocks.append((
            WriteAsciiBlock('.log.txt'), [1], []))
        Pipeline(blocks).main()
        test_byte = open('.log.txt', 'r').read().split(' ')
        test_nums = np.array([float(x) for x in test_byte])
        self.assertLess(np.max(test_nums), 256)
        self.assertEqual(test_nums.size, 12800)
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
        number_fftd = len(open(self.logfile, 'r').read().split('\n'))
        number_fftd = np.loadtxt(self.logfile).size
        open(self.logfile, 'w').close()
        ## Run pipeline again with simple copy
        self.blocks[1] = (CopyBlock(), [0], [1])
        Pipeline(self.blocks).main()
        number_copied = np.loadtxt(self.logfile).size
        self.assertAlmostEqual(number_fftd, 2*number_copied)
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
    def test_simple_ifft(self):
        """Put test data through a ring buffer and check correctness"""
        self.logfile = '.log.txt'
        self.blocks = []
        test_array = [1, 2, 3]
        self.blocks.append((TestingBlock(test_array), [], [0]))
        self.blocks.append((IFFTBlock(gulp_size=3*4), [0], [1]))
        self.blocks.append((WriteAsciiBlock(self.logfile), [1], []))
        open(self.logfile, 'w').close()
        Pipeline(self.blocks).main()
        true_result = np.fft.ifft(test_array)
        result = np.loadtxt(self.logfile).astype(np.float32).view(np.complex64)
        np.testing.assert_almost_equal(result, true_result, 2)
    def test_equivalent_data_to_copy(self):
        """Test that the data coming out of this pipeline is equivalent
        the initial read data"""
        self.logfile = '.log.txt'
        self.blocks = []
        self.blocks.append((
            SigprocReadBlock(
                '/data1/mcranmer/data/fake/1chan8bitNoDM.fil'),
            [], [0]))
        self.blocks.append((FFTBlock(gulp_size=4096*8*8*8*8), [0], [1]))
        self.blocks.append((IFFTBlock(gulp_size=4096*8*8*8*8), [1], [2]))
        self.blocks.append((WriteAsciiBlock(self.logfile), [2], []))
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
class TestPipeline(unittest.TestCase):
    """Test rigidity and features of the pipeline"""
    def test_naming_rings(self):
        """Name the rings instead of numerating them"""
        blocks = []
        blocks.append((TestingBlock([1, 2, 3]), [], ['ring1']))
        blocks.append((WriteAsciiBlock('.log.txt', gulp_size=3*4), ['ring1'], []))
        open('.log.txt', 'w').close()
        Pipeline(blocks).main()
        result = np.loadtxt('.log.txt').astype(np.float32)
        np.testing.assert_almost_equal(result, [1, 2, 3])
    def test_pass_rings(self):
        """Pass rings entirely instead of naming/numerating them"""
        block_set_one = []
        block_set_two = []
        ring1 = Ring()
        block_set_one.append((TestingBlock([1, 2, 3]), [], [ring1]))
        block_set_two.append((WriteAsciiBlock('.log.txt', gulp_size=3*4), [ring1], []))
        open('.log.txt', 'w').close()
        Pipeline(block_set_one).main() # The ring should communicate between the pipelines
        Pipeline(block_set_two).main()
        result = np.loadtxt('.log.txt').astype(np.float32)
        np.testing.assert_almost_equal(result, [1, 2, 3])
class TestMultiTransformBlock(unittest.TestCase):
    """Test call syntax and function of a multi transform block"""
    def test_add_block(self):
        """Try some syntax on an addition block."""
        my_ring = Ring()
        blocks = []
        blocks.append([TestingBlock([1, 2]), [], [0]])
        blocks.append([TestingBlock([1, 6]), [], [1]])
        blocks.append([TestingBlock([9, 2]), [], [2]])
        blocks.append([TestingBlock([6, 2]), [], [3]])
        blocks.append([TestingBlock([1, 2]), [], [4]])
        blocks.append([
            MultiAddBlock(),
            {'in_1': 0, 'in_2':1, 'out_sum': 'first_sum'}])
        blocks.append([
            MultiAddBlock(),
            {'in_1': 2, 'in_2':3, 'out_sum': 'second_sum'}])
        blocks.append([
            MultiAddBlock(),
            {'in_1': 'first_sum', 'in_2':'second_sum', 'out_sum': 'third_sum'}])
        blocks.append([
            MultiAddBlock(),
            {'in_1': 'third_sum', 'in_2':4, 'out_sum': my_ring}])
        def assert_result_of_addition(array):
            """Make sure that the above arrays add up to what we expect"""
            np.testing.assert_almost_equal(array, [18, 14])
        blocks.append((NumpyBlock(assert_result_of_addition, outputs=0), {'in_1': my_ring}))
        Pipeline(blocks).main()
    def test_for_bad_ring_definitions(self):
        """Try to pass bad input and outputs"""
        blocks = []
        blocks.append([TestingBlock([1, 2]), [], [0]])
        blocks.append([
            MultiAddBlock(),
            {'in_2':0, 'out_sum': 1}])
        blocks.append([WriteAsciiBlock('.log.txt'), [1], []])
        with self.assertRaises(AssertionError):
            Pipeline(blocks).main()
        blocks[1] = [
            MultiAddBlock(),
            {'bad_ring_name':0, 'in_2':0, 'out_sum': 1}]
        with self.assertRaises(AssertionError):
            Pipeline(blocks).main()
class TestSplitterBlock(unittest.TestCase):
    """Test a block which splits up incoming data into two rings"""
    def test_simple_half_split(self):
        """Try to split up a single array in half, and dump to file"""
        blocks = []
        blocks.append([TestingBlock([1, 2]), [], [0]])
        blocks.append([SplitterBlock([[0], [1]]), {'in': 0, 'out_1':1, 'out_2':2}])
        blocks.append([WriteAsciiBlock('.log1.txt', gulp_size=4), [1], []])
        blocks.append([WriteAsciiBlock('.log2.txt', gulp_size=4), [2], []])
        Pipeline(blocks).main()
        first_log = np.loadtxt('.log1.txt')
        second_log = np.loadtxt('.log2.txt')
        self.assertEqual(first_log.size, 1)
        self.assertEqual(second_log.size, 1)
        np.testing.assert_almost_equal(first_log+1, second_log)
class TestNumpyBlock(unittest.TestCase):
    """Tests for a block which can call arbitrary functions that work on numpy arrays.
        This should include the many numpy, scipy and astropy functions.
        Furthermore, this block should automatically move GPU data to CPU,
        call the numpy function, and then put out data on a CPU ring.
        The purpose of this ring is mainly for tests or filling in missing
        functionality."""
    def setUp(self):
        """Set up a pipeline for a numpy operation in the middle"""
        self.blocks = []
        self.test_array = [1, 2, 3, 4]
        self.blocks.append((TestingBlock(self.test_array), [], [0]))
        self.blocks.append((WriteAsciiBlock('.log.txt'), [1], []))
        self.expected_result = []
    def tearDown(self):
        """Run the pipeline and test the output against the expectation"""
        Pipeline(self.blocks).main()
        if np.array(self.expected_result).dtype == 'complex128':
            result = np.loadtxt('.log.txt', dtype=np.float64).view(np.complex128)
        else:
            result = np.loadtxt('.log.txt').astype(np.float32)
        np.testing.assert_almost_equal(result, self.expected_result)
    def test_simple_copy(self):
        """Perform a np.copy on a ring"""
        self.blocks.append([
            NumpyBlock(function=np.copy),
            {'in_1': 0, 'out_1': 1}])
        self.expected_result = [1, 2, 3, 4]
    def test_boolean_output(self):
        """Convert a ring into boolean output"""
        def greater_than_two(array):
            """Return a matrix representing whether each element
                is greater than 2"""
            return array > 2
        self.blocks.append([
            NumpyBlock(function=greater_than_two),
            {'in_1': 0, 'out_1': 1}])
        self.expected_result = [0, 0, 1, 1]
    def test_different_size_output(self):
        """Test that the output size can be different"""
        def first_half(array):
            """Only return the first half of the input vector"""
            array = np.array(array)
            return array[:int(array.size/2)]
        self.blocks.append([
            NumpyBlock(function=first_half),
            {'in_1': 0, 'out_1': 1}])
        self.expected_result = first_half(self.test_array)
    def test_complex_output(self):
        """Test that complex data can be generated"""
        self.blocks.append([
            NumpyBlock(function=np.fft.fft),
            {'in_1': 0, 'out_1': 1}])
        self.expected_result = np.fft.fft(self.test_array)
    def test_two_inputs(self):
        """Test that two input rings work"""
        def dstack_handler(array_1, array_2):
            """Stack two matrices along a third dimension"""
            return np.dstack((array_1, array_2))
        self.blocks.append([
            NumpyBlock(function=np.copy),
            {'in_1': 0, 'out_1': 2}])
        self.blocks.append([
            NumpyBlock(function=dstack_handler, inputs=2),
            {'in_1': 0, 'in_2': 2, 'out_1': 1}])
        self.expected_result = np.dstack((self.test_array, self.test_array)).ravel()
    def test_100_inputs(self):
        """Test that 100 input rings work"""
        def dstack_handler(*args):
            """Stack all input arrays"""
            return np.dstack(tuple(args))
        number_inputs = 100
        connections = {'in_1': 0, 'out_1': 1}
        for index in range(number_inputs):
            self.blocks.append([
                NumpyBlock(function=np.copy),
                {'in_1': 0, 'out_1': index+2}])
            connections['in_'+str(index+2)] = index+2
        self.blocks.append([
            NumpyBlock(function=dstack_handler, inputs=len(connections)-1),
            connections])
        self.expected_result = np.dstack((self.test_array,)*(len(connections)-1)).ravel()
    def test_two_outputs(self):
        """Test that two output rings work by copying input data to both"""
        def double(array):
            """Return two of the inputted matrix"""
            return (array, array)
        self.blocks.append([
            NumpyBlock(function=double, outputs=2),
            {'in_1': 0, 'out_1': 2, 'out_2': 1}])
        self.expected_result = [1, 2, 3, 4]
    def test_10_input_10_output(self):
        """Test that 10 input and 10 output rings work"""
        def dstack_handler(*args):
            """Stack all input arrays"""
            return np.dstack(tuple(args))
        def identity(*args):
            """Return all arrays passed"""
            return args
        number_rings = 10
        connections = {}
        for index in range(number_rings):
            #Simple 1 to 1 copy block
            self.blocks.append([
                NumpyBlock(function=np.copy),
                {'in_1': 0, 'out_1': index+2}])
            connections['in_'+str(index+1)] = index+2
            connections['out_'+str(index+1)] = index+2+number_rings
        #Copy all inputs to all outputs
        self.blocks.append([
            NumpyBlock(function=identity, inputs=number_rings, outputs=number_rings),
            dict(connections)])
        second_connections = {}
        for key in connections:
            if key[:3] == 'out':
                second_connections['in'+key[3:]] = int(connections[key])
        second_connections['out_1'] = 1
        #Stack N input rings into 1 output ring
        self.blocks.append([
            NumpyBlock(function=dstack_handler, inputs=number_rings, outputs=1),
            second_connections])
        self.expected_result = np.dstack((self.test_array,)*(len(second_connections)-1)).ravel()
    def test_zero_outputs(self):
        """Test zero outputs on NumpyBlock. Nothing should be sent through self.function at init"""
        def assert_something(array):
            """Assert the array is only 4 numbers, and return nothing"""
            np.testing.assert_almost_equal(array, [1, 2, 3, 4])
        self.blocks.append([
            NumpyBlock(function=assert_something, outputs=0),
            {'in_1': 0}])
        self.blocks.append([
            NumpyBlock(function=np.copy, outputs=1),
            {'in_1': 0, 'out_1': 1}])
        self.expected_result = [1, 2, 3, 4]
    def test_global_variable_capture(self):
        """Test that we can pull out a number from a ring using NumpyBlock"""
        self.global_variable = np.array([])
        def create_global_variable(array):
            """Try to append the array to a global variable"""
            self.global_variable = np.copy(array)
        self.blocks.append([
            NumpyBlock(function=create_global_variable, outputs=0),
            {'in_1': 0}])
        self.blocks.append([
            NumpyBlock(function=np.copy),
            {'in_1': 0, 'out_1': 1}])
        Pipeline(self.blocks).main()
        open('.log.txt', 'w').close()
        np.testing.assert_almost_equal(self.global_variable, [1, 2, 3, 4])
        self.expected_result = [1, 2, 3, 4]
class TestNumpySourceBlock(unittest.TestCase):
    """Tests for a block which can call arbitrary functions that work on numpy arrays.
        This should include the many numpy, scipy and astropy functions.
        Furthermore, this block should automatically move GPU data to CPU,
        call the numpy function, and then put out data on a CPU ring.
        The purpose of this ring is mainly for tests or filling in missing
        functionality."""
    def setUp(self):
        """Set up some parameters that every test uses"""
        self.occurences = 0
    def test_simple_single_generation(self):
        """For single yields, should act like a TestingBlock"""
        def generate_one_array():
            """Put out a single numpy array"""
            yield np.array([1, 2, 3, 4]).astype(np.float32)
        def assert_expectation(array):
            """Assert the array is as expected"""
            np.testing.assert_almost_equal(array, [1, 2, 3, 4])
            self.occurences += 1
        blocks = []
        blocks.append((NumpySourceBlock(generate_one_array), {'out_1': 0}))
        blocks.append((NumpyBlock(assert_expectation, outputs=0), {'in_1': 0}))
        Pipeline(blocks).main()
        self.assertEqual(self.occurences, 1)
    def test_multiple_yields(self):
        """Should be able to repeat generation of an array"""
        def generate_10_arrays():
            """Put out 10 numpy arrays"""
            for i in range(10):
                yield np.array([1, 2, 3, 4]).astype(np.float32)
        def assert_expectation(array):
            """Assert the array is as expected"""
            np.testing.assert_almost_equal(array, [1, 2, 3, 4])
            self.occurences += 1
        blocks = []
        blocks.append((NumpySourceBlock(generate_10_arrays), {'out_1': 0}))
        blocks.append((NumpyBlock(assert_expectation, outputs=0), {'in_1': 0}))
        Pipeline(blocks).main()
        self.assertEqual(self.occurences, 10)
    def test_multiple_output_rings(self):
        """Multiple output ring test."""
        def generate_many_arrays():
            """Put out 10x10 numpy arrays"""
            for _ in range(10):
                yield (np.array([1, 2, 3, 4]).astype(np.float32),)*10
        def assert_expectation(*args):
            """Assert the arrays are as expected"""
            assert len(args) == 10
            for array in args:
                np.testing.assert_almost_equal(array, [1, 2, 3, 4])
            self.occurences += 1
        blocks = []
        blocks.append((
            NumpySourceBlock(generate_many_arrays, outputs=10),
            {'out_%d'%(i+1):i for i in range(10)}))
        blocks.append((
            NumpyBlock(assert_expectation, inputs=10, outputs=0),
            {'in_%d'%(i+1):i for i in range(10)}))
        Pipeline(blocks).main()
        self.assertEqual(self.occurences, 10)
    def test_different_types(self):
        """Try to output different type arrays"""
        def generate_different_type_arrays():
            """Put out arrays of different types"""
            arrays = []
            for array_type in ['float32', 'float64', 'int8', 'uint8']:
                numpy_type = np.dtype(array_type).type
                arrays.append(np.array([1, 2, 3, 4]).astype(numpy_type))
            arrays.append(np.array([1+10j]))
            yield arrays
        def assert_expectation(*args):
            """Assert the arrays are as expected"""
            self.occurences += 1
            self.assertEqual(len(args), 5)
            for index, array_type in enumerate(['float32', 'float64', 'int8', 'uint8']):
                self.assertTrue(str(args[index].dtype) == array_type)
                np.testing.assert_almost_equal(args[index], [1, 2, 3, 4])
            np.testing.assert_almost_equal(args[-1], np.array([1+10j]))
        blocks = []
        blocks.append((
            NumpySourceBlock(generate_different_type_arrays, outputs=5),
            {'out_%d'%(i+1):i for i in range(5)}))
        blocks.append((
            NumpyBlock(assert_expectation, inputs=5, outputs=0),
            {'in_%d'%(i+1):i for i in range(5)}))
        Pipeline(blocks).main()
        self.assertEqual(self.occurences, 1)
    def test_header_output(self):
        """Output a header for a ring explicitly"""
        def generate_array_and_header():
            """Output the desired header of an array"""
            header = {'dtype': 'complex128', 'nbit': 128}
            yield np.array([1, 2, 3, 4]), header
        def assert_expectation(array):
            "Assert that the array has a complex datatype"
            np.testing.assert_almost_equal(array, [1, 2, 3, 4])
            self.assertEqual(array.dtype, np.dtype('complex128'))
            self.occurences += 1
        blocks = []
        blocks.append((
            NumpySourceBlock(generate_array_and_header, grab_headers=True),
            {'out_1': 0}))
        blocks.append((NumpyBlock(assert_expectation, outputs=0), {'in_1': 0}))
        Pipeline(blocks).main()
        self.assertEqual(self.occurences, 1)
    def test_multi_header_output(self):
        """Output multiple arrays and headers to fill up rings"""
        def generate_array_and_header():
            """Output the desired header of an array"""
            header_1 = {'dtype': 'complex128', 'nbit': 128}
            header_2 = {'dtype': 'complex64', 'nbit': 64}
            yield (
                np.array([1, 2, 3, 4]), header_1,
                np.array([1, 2]), header_2)
        def assert_expectation(array1, array2):
            "Assert that the arrays have different complex datatypes"
            np.testing.assert_almost_equal(array1, [1, 2, 3, 4])
            np.testing.assert_almost_equal(array2, [1, 2])
            self.assertEqual(array1.dtype, np.dtype('complex128'))
            self.assertEqual(array2.dtype, np.dtype('complex64'))
            self.occurences += 1
        blocks = []
        blocks.append((
            NumpySourceBlock(generate_array_and_header, outputs=2, grab_headers=True),
            {'out_1': 0, 'out_2': 1}))
        blocks.append((NumpyBlock(assert_expectation, inputs=2, outputs=0), {'in_1': 0, 'in_2': 1}))
        Pipeline(blocks).main()
        self.assertEqual(self.occurences, 1)
        #TODO: Add tests for changing output of generator
        #TODO: Add test for Pipeline calling _main, which sets core.
