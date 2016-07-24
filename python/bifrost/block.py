"""@package block This file defines a generic block class.

Right now the only possible block type is one
of a simple transform which works on a span by span basis.
"""
import json
import threading
import numpy as np
import matplotlib
## Use a graphical backend which supports threading
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import bifrost
from bifrost import affinity
from bifrost.ring import Ring
from bifrost.sigproc import SigprocFile, unpack

class Pipeline(object):
    """Class which connects blocks linearly, with
        one ring between each block. Does this by creating
        one ring for each input/output 'port' of each block,
        and running data through the rings."""
    def __init__(self, blocks):
        self.blocks = blocks
        self.rings = {}
        for index in self.unique_ring_names():
            if isinstance(index, Ring):
                self.rings[str(index)] = index
            else:
                self.rings[index] = Ring()
    def unique_ring_names(self):
        """Return a list of unique ring indices"""
        all_names = []
        for block in self.blocks:
            for port in block[1:]:
                for index in port:
                    if isinstance(index, Ring):
                        all_names.append(index)
                    else:
                        all_names.append(str(index))
        return set(all_names)
    def main(self):
        """Start the pipeline, and finish when all threads exit"""
        threads = []
        for block in self.blocks:
            input_rings = []
            output_rings = []
            input_rings.extend(
                [self.rings[str(ring_index)] for ring_index in block[1]])
            output_rings.extend(
                [self.rings[str(ring_index)] for ring_index in block[2]])
            if issubclass(type(block[0]), SourceBlock):
                threads.append(threading.Thread(
                    target=block[0].main,
                    args=[output_rings[0]]))
            elif issubclass(type(block[0]), SinkBlock):
                threads.append(threading.Thread(
                    target=block[0].main,
                    args=[input_rings[0]]))
            else:
                threads.append(threading.Thread(
                    target=block[0].main,
                    args=[input_rings, output_rings]))
        for thread in threads:
            thread.daemon = True
            thread.start()
        for thread in threads:
            # Wait for exit
            thread.join()

class TransformBlock(object):
    """Defines the structure for a transform block"""
    def __init__(self, gulp_size=4096):
        super(TransformBlock, self).__init__()
        self.gulp_size = gulp_size
        self.out_gulp_size = None
        self.input_header = {}
        self.output_header = {}
        self.core = -1
    def load_settings(self, input_header):
        """Load in input header and set up block attributes
        @param[in] input_header Header sent from input ring"""
        self.output_header = input_header
    def iterate_ring_read(self, input_ring):
        """Iterate through one input ring"""
        input_ring.resize(self.gulp_size)
        for sequence in input_ring.read(guarantee=True):
            self.load_settings(sequence.header)
            for span in sequence.read(self.gulp_size):
                yield span
    def iterate_ring_write(
            self, output_ring, sequence_name="",
            sequence_time_tag=0, sequence_nringlet=1):
        """Iterate through one output ring"""
        if self.out_gulp_size is None:
            self.out_gulp_size = self.gulp_size
        output_ring.resize(self.out_gulp_size)
        with output_ring.begin_writing() as oring:
            with oring.begin_sequence(
                sequence_name, sequence_time_tag,
                header=self.output_header,
                nringlet=sequence_nringlet) as oseq:
                with oseq.reserve(self.out_gulp_size) as span:
                    yield span
    def ring_transfer(self, input_ring, output_ring):
        """Iterate through two rings span-by-span"""
        input_ring.resize(self.gulp_size)
        for sequence in input_ring.read(guarantee=True):
            self.load_settings(sequence.header)
            if self.out_gulp_size is None:
                self.out_gulp_size = self.gulp_size
            output_ring.resize(self.out_gulp_size)
            with output_ring.begin_writing() as oring:
                with oring.begin_sequence(
                    sequence.name, sequence.time_tag,
                    header=self.output_header,
                    nringlet=sequence.nringlet) as oseq:
                    for ispan in sequence.read(self.gulp_size):
                        with oseq.reserve(ispan.size*self.out_gulp_size/self.gulp_size) as ospan:
                            yield ispan, ospan

class SourceBlock(object):
    """Defines the structure for a source block"""
    def __init__(self, gulp_size=4096):
        super(SourceBlock, self).__init__()
        self.gulp_size = gulp_size
        self.output_header = {}
        self.core = -1
    def iterate_ring_write(
            self, output_ring, sequence_name="",
            sequence_time_tag=0):
        """Iterate over output ring
        @param[in] output_ring Ring to write to
        @param[in] sequence_name Name to label sequence
        @param[in] sequence_time_tag Time tag to label sequence
        """
        output_ring.resize(self.gulp_size)
        with output_ring.begin_writing() as oring:
            with oring.begin_sequence(
                sequence_name, sequence_time_tag,
                header=self.output_header,
                nringlet=1) as oseq:
                with oseq.reserve(self.gulp_size) as span:
                    yield span
class SinkBlock(object):
    """Defines the structure for a transform block"""
    def __init__(self, gulp_size=4096):
        super(SinkBlock, self).__init__()
        self.gulp_size = gulp_size
        self.header = {}
        self.core = -1
    def load_settings(self, input_header):
        """Load in settings from input ring header"""
        self.header = json.loads(input_header.tostring())
    def iterate_ring_read(self, input_ring):
        """Iterate through one input ring
        @param[in] input_ring Ring to read through"""
        input_ring.resize(self.gulp_size)
        for sequence in input_ring.read(guarantee=True):
            self.load_settings(sequence.header)
            for span in sequence.read(self.gulp_size):
                yield span
class TestingBlock(SourceBlock):
    """Block for debugging purposes.
    Allows you to pass arbitrary N-dimensional arrays in initialization,
    which will be outputted into a ring buffer"""
    def __init__(self, test_array):
        """@param[in] test_array A list or numpy array containing test data"""
        super(TestingBlock, self).__init__()
        self.test_array = np.array(test_array).astype(np.float32)
        self.output_header = json.dumps(
            {'nbit':32,
             'dtype':str(np.float32),
             'shape':self.test_array.shape})
    def main(self, output_ring):
        """Put the test array onto the output ring
        @param[in] output_ring Holds the flattend test array in a single span"""
        self.gulp_size = self.test_array.nbytes
        for ospan in self.iterate_ring_write(output_ring):
            ospan.data_view(np.float32)[0][:] = self.test_array.ravel()
            break
class WriteHeaderBlock(SinkBlock):
    """Prints the header of a ring to a file"""
    def __init__(self, filename):
        """@param[in] test_array A list or numpy array containing test data"""
        super(WriteHeaderBlock, self).__init__()
        self.filename = filename
    def load_settings(self, input_header):
        """Load the header from json
        @param[in] input_header The header from the ring"""
        write_file = open(self.filename, 'w')
        write_file.write(str(json.loads(input_header.tostring())))
    def main(self, input_ring):
        """Put the header into the file
        @param[in] input_ring Contains the header in question"""
        self.gulp_size = 1
        span_dummy_generator = self.iterate_ring_read(input_ring)
        span_dummy_generator.next()
class FFTBlock(TransformBlock):
    """Performs complex to complex IFFT on input ring data"""
    def __init__(self, gulp_size):
        super(FFTBlock, self).__init__()
    def load_settings(self, input_header):
        header = json.loads(input_header.tostring())
        self.out_gulp_size = self.gulp_size*64/header['nbit']
        self.nbit = header['nbit']
        self.dtype = np.dtype(header['dtype'].split()[1].split(".")[1].split("'")[0]).type
        header['nbit'] = 64
        header['dtype'] = str(np.complex64)
        self.output_header = json.dumps(header)
    def main(self, input_rings, output_rings):
        """
        @param[in] input_rings First ring in this list will be used for
            data
        @param[out] output_rings First ring in this list will be used for 
            data output."""
        for ispan, ospan in self.ring_transfer(input_rings[0], output_rings[0]):
            if self.nbit < 8:
                unpacked_data = unpack(ispan.data_view(self.dtype), self.nbit)
            else:
                unpacked_data = ispan.data_view(self.dtype)
            result = np.fft.fft(unpacked_data.astype(np.float32))
            ospan.data_view(np.complex64)[0][:] = result[0][:]
class IFFTBlock(TransformBlock):
    """Performs complex to complex IFFT on input ring data"""
    def __init__(self, gulp_size):
        super(IFFTBlock, self).__init__()
        self.gulp_size = gulp_size
    def load_settings(self, input_header):
        header = json.loads(input_header.tostring())
        self.out_gulp_size = self.gulp_size*64/header['nbit']
        self.nbit = header['nbit']
        self.dtype = np.dtype(header['dtype'].split()[1].split(".")[1].split("'")[0]).type
        header['nbit'] = 64
        header['dtype'] = str(np.complex64)
        self.output_header = json.dumps(header)
    def main(self, input_rings, output_rings):
        """
        @param[in] input_rings First ring in this list will be used for
            data
        @param[out] output_rings First ring in this list will be used for 
            data output."""
        for ispan, ospan in self.ring_transfer(input_rings[0], output_rings[0]):
            if self.nbit < 8:
                unpacked_data = unpack(ispan.data_view(self.dtype), self.nbit)
            else:
                unpacked_data = ispan.data_view(self.dtype)
            result = np.fft.ifft(unpacked_data)
            ospan.data_view(np.complex64)[0][:] = result[0][:]
class WriteAsciiBlock(SinkBlock):
    """Copies input ring's data into ascii format
        in a text file."""
    def __init__(self, filename, gulp_size=1048576):
        """@param[in] filename Name of file to write ascii to
        @param[out] gulp_size How much of the file to write at once"""
        super(WriteAsciiBlock, self).__init__()
        self.filename = filename
        self.gulp_size = gulp_size 
        self.dtype = np.uint8
        ## erase file
        open(self.filename, "w").close()
    def load_settings(self, input_header):
        header_dict = json.loads(input_header.tostring())
        self.nbit = header_dict['nbit']
        self.dtype = np.dtype(header_dict['dtype'].split()[1].split(".")[1].split("'")[0]).type
    def main(self, input_ring):
        """Initiate the writing to filename
        @param[in] input_rings First ring in this list will be used for
            data
        @param[out] output_rings This list of rings won't be used."""
        span_generator = self.iterate_ring_read(input_ring)
        for span in span_generator:
            if self.nbit < 8:
                unpacked_data = unpack(span.data_view(self.dtype), self.nbit)
            else:
                unpacked_data = span.data_view(self.dtype)
            text_file = open(self.filename, 'a')
            if self.dtype == np.complex64:
                np.savetxt(text_file, span.data_view(self.dtype).view(np.float32))
            else:
                np.savetxt(text_file, unpacked_data)
class CopyBlock(TransformBlock):
    """Copies input ring's data to the output ring"""
    def __init__(self, gulp_size=1048576):
        super(CopyBlock, self).__init__(gulp_size=gulp_size)
    def main(self, input_rings, output_rings):
        input_ring = input_rings[0]
        for output_ring in output_rings:
            for ispan, ospan in self.ring_transfer(input_ring, output_ring):
                bifrost.memory.memcpy2D(ospan.data, ispan.data)
class SigprocReadBlock(SourceBlock):
    """This block reads in a sigproc filterbank
    (.fil) file into a ring buffer"""
    def __init__(
            self, filename,
            gulp_nframe=4096, core=-1):
        """
        @param[in] filename filterbank file to read
        @param[in] gulp_nframe Time samples to read
            in at a time
        @param[in] core Which CPU core to bind to (-1) is
            any
        """
        super(SigprocReadBlock, self).__init__()
        self.filename = filename
        self.gulp_nframe = gulp_nframe
        self.core = core
    def main(self, output_ring):
        """Read in the sigproc file to output_ring
        @param[in] output_ring Ring to write to"""
        with SigprocFile().open(self.filename,'rb') as ifile:
            ifile.read_header()
            ohdr = {}
            ohdr['frame_shape'] = (ifile.nchans, ifile.nifs)
            ohdr['frame_size'] = ifile.nchans*ifile.nifs
            ohdr['frame_nbyte'] = ifile.nchans*ifile.nifs*ifile.nbits/8
            ohdr['frame_axes'] = ('pol', 'chan')
            ohdr['ringlet_shape'] = (1,)
            ohdr['ringlet_axes'] = ()
            ohdr['dtype'] = str(ifile.dtype)
            ohdr['nbit'] = ifile.nbits
            ohdr['tsamp'] = float(ifile.header['tsamp'])
            ohdr['tstart'] = float(ifile.header['tstart'])
            ohdr['fch1'] = float(ifile.header['fch1'])
            ohdr['foff'] = float(ifile.header['foff'])
            self.output_header = json.dumps(ohdr)
            self.gulp_size = self.gulp_nframe*ifile.nchans*ifile.nifs*ifile.nbits/8
            out_span_generator = self.iterate_ring_write(output_ring)
            for span in out_span_generator:
                size = ifile.file_object.readinto(span.data.data)
                span.commit(size)
                if size == 0:
                    break

class KurtosisBlock(TransformBlock):
    """This block performs spectral kurtosis and cleaning
        on sigproc-formatted data in rings"""
    def __init__(self, gulp_size=1048576, core=-1):
        """
        @param[in] input_ring Ring containing a 1d
            timeseries
        @param[out] output_ring Ring will contain a 1d
            timeseries that will be cleaned of RFI
        @param[in] core Which OpenMP core to use for
            this block. (-1 is any)
        """
        super(KurtosisBlock, self).__init__()
        self.SAMPLING_RATE = 41.66666666667e-6    # Should pass in header
        self.N_CHAN = 109
        self.EXPECTED_V2 = 0.5      # Just use for testing
        self.gulp_size = gulp_size
        self.core = core
    def load_settings(self, input_header):
        self.output_header = input_header
        self.settings = json.loads(input_header.tostring())
        self.nchan = self.settings["frame_shape"][0]
        #TODO: Clean this up
        dtype_str = self.settings["dtype"].split()[1].split(".")[1].split("'")[0]
        self.dtype = np.dtype(dtype_str)
    def main(self, input_rings, output_rings):
        """Calls a kurtosis algorithm and uses the result
            to clean the input data of RFI, and move it to the
            output ring."""
        for ispan, ospan in self.ring_transfer(input_rings[0], output_rings[0]):
            nsample = ispan.size/self.nchan/(self.settings['nbit']/8)
            power = ispan.data.reshape(
                nsample, 
                self.nchan*self.settings['nbit']/8).view(self.dtype) # Raw data -> power array of the right type
            # Follow section 3.1 of the Nita paper. 
            M = power.shape[0] # Number of samples, the sample is a power value in a frequency bin from an FFT, i.e. the beamformer values in a channel
            bad_channels = []
            for chan in range(self.nchan):
                S1 = np.sum(power[:, chan])
                S2 = np.sum(power[:, chan]**2)
                V2 = (M/(M-1))*(M*S2/(S1**2) -1)          # Equation 21
                VarV2 = 24.0/M                            # Equation 23
                if abs(self.EXPECTED_V2-V2) > 0.1:
                    bad_channels.append(chan)
            if len(bad_channels) > 0:
                flag_power = power.copy()
                for chan in range(self.nchan):
                    if chan in bad_channels: flag_power[:, chan] = 0    # set bad channel to zero
                ospan.data[0][:] = flag_power.view(dtype=np.uint8).ravel()
            else:
                with oseq.reserve(ispan.size) as ospan:
                    bifrost.memory.memcpy2D(ospan.data, ispan.data)      # Transfer data unchanged

class DedisperseBlock(object):
    """This block calculates the dedispersion of 
        sigproc-formatted data in a ring, and tags
        it in the headers"""
    def __init__(
            self, ring, core=-1,
            gulp_size=4096):
        """
        @param[in] ring Ring containing a 1d
            timeseries with a source affected
            by DM
        @param[in] core Which OpenMP core to use for 
            this block. (-1 is any)
        @param[in] gulp_size How many bytes of the ring to
            read at once.
        """
        self.ring = ring
        self.core = core
        self.gulp_size = gulp_size
    def main(self):
        """Initiate the block's processing"""
        affinity.set_core(self.core)
        self.dedisperse()
    def dedisperse(self):
        """Dedisperse on input ring, tagging the output ring.
        This dedispersion algorithm simply adjusts the start
        time for each channel in the header."""
        pass

class FoldBlock(TransformBlock):
    """This block folds a signal into a histogram"""
    def __init__(
            self, bins, period=1e-3, 
            gulp_size=4096*256, dispersion_measure=0,
            core=-1):
        """
        @param[in] bins The total number of bins to fold into
        @param[in] period Period to fold over (s)
        @param[in] gulp_size How many bytes of the ring to
            read at once.
        @param[in] dispersion_measure DM of the desired
            source (pc cm^-3)
        @param[in] core Which OpenMP core to use for 
            this block. (-1 is any)
        """
        super(FoldBlock, self).__init__()
        self.bins = bins
        self.gulp_size = gulp_size
        self.period = period
        self.dispersion_measure = dispersion_measure 
        self.core = core
    def calculate_bin_indices(
            self, tstart, tsamp, data_size):
        """Calculate the bin that each time sample should be
            added to
        @param[in] tstart Time of the first element (s)
        @param[in] tsamp Difference between the times of
            consecutive elements (s)
        @param[in] data_size Number of elements
        @return Which bin each sample is folded into
        """
        arrival_time = tstart+tsamp*np.arange(data_size)
        phase = np.fmod(arrival_time, self.period)
        return np.floor(phase/self.period*self.bins).astype(int)
    def insert_zeros_evenly(self, input_data, number_zeros):
        """Insert zeros as elements in input_data.
            These zeros are distibuted evenly throughout
            the function, to help for binning of oddly
            shaped arrays.
        @param[in] input_data 1D array to contain zeros.
        @param[out] number_zeros Number of zeros that need
            to be added.
        @returns input_data with extra zeros"""
        insert_index = np.floor(
            np.arange(
                number_zeros,
                step=1.0)*float(input_data.size)/number_zeros)
        output_data = np.insert(
            input_data, insert_index,
            np.zeros(number_zeros))
        return output_data
    def calculate_delay(self, frequency, reference_frequency):
        """Calculate the time delay because of frequency dispersion
        @param[in] frequency The current channel's frequency(MHz)
        @param[in] reference_frequency The frequency of the 
            channel we will hold at zero time delay(MHz)"""
        frequency_factor = \
            np.power(reference_frequency/1000, -2) -\
            np.power(frequency/1000, -2)
        return 4.15e-3*self.dispersion_measure*frequency_factor
    def load_settings(self, input_header):
        self.data_settings = json.loads(
            "".join(
                [chr(item) for item in input_header]))
        self.output_header = json.dumps({'nbit':32, 'dtype': str(np.float32)})
    def main(self, input_rings, output_rings):
        """Generate a histogram from the input ring data
        @param[in] input_rings List with first ring containing
            data of interest. Must terminate before histogram
            is generated.
        @param[out] output_rings First ring in this list
            will contain the output histogram"""
        histogram = np.reshape(
            np.zeros(self.bins).astype(np.float32),
            (1, self.bins))
        iterates = 0
        for span in self.iterate_ring_read(input_rings[0]):
            nchans = self.data_settings['frame_shape'][0]
            tstart = self.data_settings['tstart']
            tstart += iterates*self.data_settings['tsamp']*self.gulp_size*8/self.data_settings['nbit']/nchans
            iterates += 1
            frequency = self.data_settings['fch1']
            for chan in range(nchans):
                modified_tstart = tstart - self.calculate_delay(
                    frequency,
                    self.data_settings['fch1'])
                frequency -= self.data_settings['foff']
                sort_indices = np.argsort(
                    self.calculate_bin_indices(
                        modified_tstart, self.data_settings['tsamp'], 
                        span.data.shape[1]/nchans))
                sorted_data = span.data[0][chan::nchans][sort_indices]
                extra_elements = np.round(self.bins*(1-np.modf(
                    float(span.data.shape[1]/nchans)/self.bins)[0])).astype(int)
                sorted_data_with_zeros = self.insert_zeros_evenly(
                    sorted_data, extra_elements)
                histogram += np.sum(
                    sorted_data_with_zeros.reshape(self.bins, -1), 1).astype(np.float32)
        self.out_gulp_size = self.bins*4
        out_span_generator = self.iterate_ring_write(output_rings[0])
        out_span = out_span_generator.next()
        bifrost.memory.memcpy(
            out_span.data_view(dtype=np.float32),
            histogram)

class WaterfallBlock(object):
    """This block creates a waterfall block
        based on the data in a ring, and stores it 
        in the headers"""
    def __init__(
            self, ring, imagename,
            core=-1, gulp_nframe=4096):
        """
        @param[in] ring Ring containing a multichannel
            timeseries
        @param[in] imagename Filename to store the
            waterfall image
        @param[in] core Which OpenMP core to use for
            this block. (-1 is any)
        @param[in] gulp_size How many bytes of the ring to
            read at once.
        """
        self.ring = ring
        self.imagename = imagename
        self.core = core
        self.gulp_nframe = gulp_nframe
        self.header = {}
    def main(self):
        """Initiate the block's processing"""
        affinity.set_core(self.core)
        waterfall_matrix = self.generate_waterfall_matrix()
        self.save_waterfall_plot(waterfall_matrix)
    def save_waterfall_plot(self, waterfall_matrix):
        """Save an image of the waterfall plot using
            thread-safe backend for pyplot, and labelling
            the plot using the header information from the
            ring
        @param[in] waterfall_matrix x axis is frequency and 
            y axis is time. Values should be power.
            """
        plt.ioff()
        print "Interactive mode off"
        print waterfall_matrix.shape
        fig = pylab.figure()
        ax = fig.gca()
        header = self.header
        ax.set_xticks(
            np.arange(0, 1.33, 0.33)*waterfall_matrix.shape[1])
        ax.set_xticklabels(
            header['fch1']-np.arange(0,4)*header['foff'])
        ax.set_xlabel("Frequency [MHz]")
        ax.set_yticks(
            np.arange(0, 1.125, 0.125)*waterfall_matrix.shape[0])
        ax.set_yticklabels(
            header['tstart']+header['tsamp']*np.arange(0, 1.125, 0.125)*waterfall_matrix.shape[0])
        ax.set_ylabel("Time (s)")
        plt.pcolormesh(
            waterfall_matrix, axes=ax, figure=fig)
        fig.autofmt_xdate()
        fig.savefig(
            self.imagename, bbox_inches='tight')
        plt.close(fig)
    def generate_waterfall_matrix(self):
        """Create a matrix for a waterfall image
            based on the ring's data"""
        waterfall_matrix = None
        self.ring.resize(self.gulp_nframe)
        # Generate a waterfall matrix:
        for sequence in self.ring.read(guarantee=True):
            ## Get the sequence's header as a dictionary
            self.header = json.loads(
                "".join(
                    [chr(item) for item in sequence.header]))
            tstart = self.header['tstart']
            tsamp = self.header['tsamp']
            nchans = self.header['frame_shape'][0]
            gulp_size = self.gulp_nframe*nchans*self.header['nbit']
            waterfall_matrix = np.zeros(shape=(0, nchans))
            print tstart, tsamp, nchans
            for span in sequence.read(gulp_size):
                array_size = span.data.shape[1]/nchans
                frequency = self.header['fch1']
                try: 
                    curr_data = np.reshape(
                        span.data,(-1, nchans))
                    waterfall_matrix = np.concatenate(
                        (waterfall_matrix, curr_data), 0)
                except:
                    print "Bad shape for waterfall"
                    pass
        return waterfall_matrix

