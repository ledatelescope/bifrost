"""@package block
This file defines a generic block class.

Right now the only possible block type is one
of a simple transform which works on a span by span basis.
"""
import json
import os
import threading
import numpy as np
import matplotlib
## Use a graphical backend which supports threading
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import bifrost
import bandfiles
from bifrost import affinity
from bifrost.ring import Ring
from bifrost.sigproc import SigprocFile

#TODO: How does GNU Radio handle this?
class Pipeline(object):
    """Class which connects blocks linearly, with
        one ring between each block. Does this by creating
        one ring for each input/output 'port' of each block,
        and running data through the rings."""
    def __init__(self, blocks):
        self.blocks = blocks
        self.rings = [Ring() for count in range(
            self.number_of_rings())]
    def number_of_rings(self):
        """Return how many rings will be used in
            this pipeline."""
        all_ports = [
            index for block in self.blocks for index in [
                port for port in block[1:] if port != None]]
        return len(np.unique(all_ports))
    def main(self):
        """Start the pipeline, and finish when all threads exit"""
        threads = []
        for block in self.blocks:
            function_rings = []
            function_rings.extend(
                [self.rings[ring_index] for ring_index in block[1]])
            function_rings.extend(
                [self.rings[ring_index] for ring_index in block[2]])
            threads.append(threading.Thread(
                target=block[0].main,
                args=function_rings))
        for thread in threads:
            thread.daemon = True
            thread.start()
        for thread in threads:
            # Wait for exit
            thread.join()


class TransformBlock(object):
    """Defines the structure for a transform block"""
    def __init__(self):
        super(TransformBlock, self).__init__()
    def main(self, input_ring, output_ring):
        """Initiate the block's transform between
            rings."""
        pass
class WriteAsciiBlock(object):
    """Copies input ring's data into ascii format
        in a text file"""
    def __init__(self, filename):
        super(WriteAsciiBlock, self).__init__()
        self.filename = filename
        ## erase file
        open(self.filename, "w").close()
    def main(self, data_ring):
        """Initiate the writing to filename"""
        gulp_size = 1048576
        data_ring.resize(gulp_size)
        for iseq in data_ring.read(guarantee=True):
            for ispan in iseq.read(gulp_size):
                np.savetxt(self.filename, ispan.data, '%d')
class CopyBlock(TransformBlock):
    """Copies input ring's data to the output ring"""
    def __init__(self, gulp_size=1048576):
        super(CopyBlock, self).__init__()
        self.inputs = 1
        self.outputs = 1
        self.gulp_size = gulp_size
    def main(self, input_ring, output_ring):
        input_ring.resize(self.gulp_size)
        output_ring.resize(self.gulp_size)
        with output_ring.begin_writing() as oring:
            for iseq in input_ring.read(guarantee=True):
                with oring.begin_sequence(
                    iseq.name, iseq.time_tag,
                    header=iseq.header,
                    nringlet=iseq.nringlet) as oseq:
                    for ispan in iseq.read(self.gulp_size):
                        with oseq.reserve(ispan.size) as ospan:
                            bifrost.memory.memcpy2D(
                                ospan.data, ispan.data)

class SigprocReadBlock(object):
    """This block reads in a sigproc filterbank
    (.fil) file into a ring buffer"""
    def __init__(
            self, filenames,
            gulp_nframe=4096, max_frames=None,
            core=-1):
        """
        @param[in] filenames filterbank files to read
        @param[in] gulp_nframe Time samples to read
            in at a time
        @param[in] max_frames Maximum samples to read from
            file (None is no max)
        @param[in] core Which CPU core to bind to (-1) is
            any
        """
        self.filenames = filenames
        self.gulp_nframe = gulp_nframe
        self.core = core
        self.max_frames = max_frames
        self.inputs = 1
    def main(self, output_ring):
        """Read in the sigproc file to output_ring"""
        affinity.set_core(self.core)
        with output_ring.begin_writing() as oring:
            for name in self.filenames:
                with SigprocFile().open(name,'rb') as ifile:
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
                    ohdr = json.dumps(ohdr)
                    gulp_nbyte = self.gulp_nframe*ifile.nchans*ifile.nifs*ifile.nbits/8
                    output_ring.resize(gulp_nbyte)
                    with oring.begin_sequence(name, header=ohdr) as osequence:
                        frames_read = 0
                        while (self.max_frames is None) or \
                                frames_read < self.max_frames:
                            with osequence.reserve(gulp_nbyte) as wspan:
                                size = ifile.file_object.readinto(wspan.data.data)
                                wspan.commit(size)
                                if size == 0:
                                    break
                                frames_read += 1


def duplicate_ring(input_ring, output_ring):
    """This function copies data between two rings
    @param[in] input_ring Ring holding data to be copied
    @param[out] output_ring Will be a copy of input_ring
    """
    gulp_size = 1048576
    input_ring.resize(gulp_size)
    output_ring.resize(gulp_size)
    with output_ring.begin_writing() as oring:
        for iseq in input_ring.read(guarantee=True):
            with oring.begin_sequence(
                iseq.name, iseq.time_tag,
                header=iseq.header,
                nringlet=iseq.nringlet) as oseq:
                for ispan in iseq.read(gulp_size):
                    with oseq.reserve(ispan.size) as ospan:
                        bifrost.memory.memcpy2D(
                            ospan.data,ispan.data)


# Read a collection of DADA files and form an array of time series data over
# many frequencies.
# TODO: Add more to the header.
# Add a list of frequencies present. This could be the full band,
# but there could be gaps. Downstream functionality has to be aware of the gaps.
# Allow specification of span size based on time interval
class DadaReadBlock(object):

  # Assemble a group of files in the time direction and the frequency direction
  # time_stamp is of the form "2016-05-24-11:04:38", or a DADA file ending in .dada
  def __init__(self, time_stamp, outring, core=-1, gulp_nframe=4096):
    self.CHANNEL_WIDTH = 0.024
    self.SAMPLING_RATE = 41.66666666667e-6
    self.N_CHAN = 109
    self.N_BEAM = 2
    self.HEADER_SIZE = 4096
    self.OBS_OFFSET = 1255680000
    self.gulp_nframe = gulp_nframe

    self.oring = outring
    self.core = core

    beamformer_scans = []
    if time_stamp[-5:] == ".dada": beamformer_scans.append(bandfiles.BandFiles(time_stamp))  # Just one file
    else:
      finish = False
      i = 0
      while not finish:         # This loop gathers files by time
        new_offset = str(i*self.OBS_OFFSET)
        file_name = time_stamp+"_"+"0"*(16-len(new_offset))+new_offset+".000000.dada"

        scan = bandfiles.BandFiles(file_name)           # The BandFiles class gathers by frequency

        finish = ( len(scan.files) == 0 )

        if not finish: beamformer_scans.append(scan)

        i += 1

    # Report what we've got
    print "Num files in time:",  len(beamformer_scans)
    print "File and number:"
    for scan in beamformer_scans:
      print os.path.basename(scan.files[0].name)+":", len(scan.files)

    self.beamformer_scans = beamformer_scans     # List of full-band time steps

  def main(self):
    bifrost.affinity.set_core(self.core)

    # Calculate some constants for sizes
    length_one_second = int(round(1/self.SAMPLING_RATE))
    ring_span_size = length_one_second*self.N_CHAN*4                    # 1 second, all the channels (109) and then 4-byte floats
    file_chunk_size = length_one_second*self.N_BEAM*self.N_CHAN*2               # 1 second, 2 beams, 109 chans, and 2 1-byte ints (real, imag)
    number_of_seconds = 120     # Change this

    ohdr = {}
    ohdr["frame_shape"] = ( self.N_CHAN, 1 )
    ohdr["nbit"] = 32
    ohdr["dtype"] = str(np.float32)
    ohdr["tstart"] = 0
    ohdr["tsamp"] = self.SAMPLING_RATE
    ohdr['foff'] = self.CHANNEL_WIDTH

    #print length_one_second, ring_span_size, file_chunk_size, number_of_chunks

    with self.oring.begin_writing() as oring:

      # Go through the files by time. 
      for scan in self.beamformer_scans:
        # Go through the frequencies
        for f in scan.files:

          print "Opening", f.name

          with open(f.name,'rb') as ifile:
            ifile.read(self.HEADER_SIZE)

            ohdr["cfreq"] = f.freq
            ohdr["fch1"] = f.freq

            self.oring.resize(ring_span_size)
            with oring.begin_sequence(f.name, header=json.dumps(ohdr)) as osequence:

              for i in range(number_of_seconds):
                  # Get a chunk of data from the file. The whole band is used, but only a chunk of time (1 second).
                  # Massage the data so it can go through the ring. That means changng the data type and flattening.
                try:
                    data = np.fromfile(ifile, count=file_chunk_size, dtype=np.int8).astype(np.float32)
                except:
                    print "Bad read. Stopping read."
                    return
                if data.size != length_one_second*self.N_BEAM*self.N_CHAN*2:
                    print "Bad data shape. Stopping read."
                    return
                data = data.reshape(length_one_second, self.N_BEAM, self.N_CHAN, 2)
                power = (data[...,0]**2 + data[...,1]**2).mean(axis=1)  # Now have time by frequency.
                print power.shape

                # Send the data
                with osequence.reserve(ring_span_size) as wspan:
                  wspan.data[0][:] = power.view(dtype=np.uint8).ravel()

def duplicate_ring(input_ring, output_ring):
    """This function copies data between two rings
    @param[in] input_ring Ring holding data to be copied
    @param[out] output_ring Will be a copy of input_ring
    """
    gulp_size = 1048576
    input_ring.resize(gulp_size)
    output_ring.resize(gulp_size)
    with output_ring.begin_writing() as oring:
        for iseq in input_ring.read(guarantee=True):
            with oring.begin_sequence(
                iseq.name, iseq.time_tag,
                header=iseq.header,
                nringlet=iseq.nringlet) as oseq:
                for ispan in iseq.read(gulp_size):
                    with oseq.reserve(ispan.size) as ospan:
                        bifrost.memory.memcpy2D(
                            ospan.data,ispan.data)

class KurtosisBlock(object):
    """This block performs spectral kurtosis and cleaning
        on sigproc-formatted data in rings"""
    def __init__(self, input_ring, output_ring, core=-1):
        """
        @param[in] input_ring Ring containing a 1d
            timeseries
        @param[out] output_ring Ring will contain a 1d
            timeseries that will be cleaned of RFI
        @param[in] core Which OpenMP core to use for
            this block. (-1 is any)
        """
        self.SAMPLING_RATE = 41.66666666667e-6    # Should pass in header
        self.N_CHAN = 109
        self.EXPECTED_V2 = 0.5      # Just use for testing

        self.iring = input_ring
        self.oring = output_ring
        self.core = core
    def main(self):
        """Initiate the block's processing"""
        affinity.set_core(self.core)
        self.rfi_clean()
    def rfi_clean(self):
        """Calls a kurtosis algorithm and uses the result
            to clean the input data of RFI, and move it to the
            output ring."""
        #duplicate_ring(self.input_ring, self.output_ring)
        length_one_second = int(round(1/self.SAMPLING_RATE))
        ring_span_size = length_one_second*self.N_CHAN*4                 # 1 second, all the channels (109) and then 4-byte floats

        self.iring.resize(ring_span_size)
        self.oring.resize(ring_span_size)

        with self.oring.begin_writing() as oring:

          for iseq in self.iring.read():

            # Have to copy the data through, so need to write as well as read       
            with oring.begin_sequence(iseq.name, header=iseq.header) as oseq:


              # Get info about the data, to form an array
              header = json.loads(iseq.header.tostring())
              nchan = header["frame_shape"][0]
              nbit = header["nbit"]
              nif = header["frame_shape"][1]
              dtype_str = header["dtype"].split()[1].split(".")[1].split("'")[0]    # Must be a better way
              dtype = np.dtype(dtype_str)
              #print iseq.name, "nchan:", nchan, "nbit:", nbit, "dtype:", dtype

              if nif != 1:
                print "Only 1 IF is supported"
                return

              for ispan in iseq.read(ring_span_size):
                # Process 1 second data
                nsample = ispan.size/nchan/(nbit/8)
                power = ispan.data.reshape(nsample, nchan*nbit/8).view(dtype)               # Raw data -> power array of the right type

                # Follow section 3.1 of the Nita paper. 
                M = power.shape[0]          # Number of samples, the sample is a power value in a frequency bin from an FFT, i.e. the beamformer values in a channel

                bad_channels = []
                for chan in range(nchan):
                  S1 = np.sum(power[:, chan])
                  S2 = np.sum(power[:, chan]**2)

                  V2 = (M/(M-1))*(M*S2/(S1**2) -1)          # Equation 21
                  VarV2 = 24.0/M                            # Equation 23

                  if abs(self.EXPECTED_V2-V2) > 0.1:
                    bad_channels.append(chan)

                if len(bad_channels) > 0:
                  flag_power = power.copy()
                  for chan in range(nchan):
                    if chan in bad_channels: flag_power[:, chan] = 0    # set bad channel to zero

                  #print "Chan flagged", bad_channels
                  with oseq.reserve(ispan.size) as ospan:
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

class FoldBlock(object):
    """This block folds a signal into a histogram"""
    def __init__(
            self, input_ring, np_output_array, 
            dispersion_measure=None, core=-1, gulp_size=4096):
        """
        @param[in] input_ring Ring containing a 1d
            timeseries
        @param[out] np_output_array Numpy array which will
            eventually contain a histogram from our folding
        @param[in] dispersion_measure DM of the desired
            source (pc cm^-3)
        @param[in] core Which OpenMP core to use for 
            this block. (-1 is any)
        @param[in] gulp_size How many bytes of the ring to
            read at once.
        """
        self.input_ring = input_ring
        self.output_array = np_output_array
        self.core = core
        self.gulp_size = gulp_size
        self.dispersion_measure = dispersion_measure
    def main(self):
        """Initiate the block's processing"""
        affinity.set_core(self.core)
        self.fold(period=1e-3, bins=100)
    def calculate_bin_indices(
        self, tstart, tsamp, data_size, period, bins):
        """Calculate the bin that each time sample should be
            added to
        @param[in] tstart Time of the first element (s)
        @param[in] tsamp Difference between the times of
            consecutive elements (s)
        @param[in] data_size Number of elements
        @param[in] period Which period to fold over (s)
        @param[in] bins The total number of bins to fold into
        @return Which bin each sample is folded into
        """
        arrival_time = tstart+tsamp*np.arange(data_size)
        phase = np.fmod(arrival_time, period)
        return np.floor(phase/period*bins).astype(int)
    def calculate_delay(self, frequency, reference_frequency):
        """Calculate the time delay because of frequency dispersion
        @param[in] frequency The current channel's frequency(MHz)
        @param[in] reference_frequency The frequency of the 
            channel we will hold at zero time delay(MHz)"""
        frequency_factor = \
            np.power(reference_frequency/1000, -2) -\
            np.power(frequency/1000, -2)
        return 4.15e-3*self.dispersion_measure*frequency_factor
    def fold(self, period, bins):
        """Fold the signal into the numpy array
        @param[in] period Period to fold over in seconds
        @param[in] bins Number of bins in the histogram
        """
        self.input_ring.resize(self.gulp_size)
        for sequence in self.input_ring.read(guarantee=True):
            ## Get the sequence's header as a dictionary
            header = json.loads(
                "".join(
                    [chr(item) for item in sequence.header]))
            tstart = header['tstart']
            tsamp = header['tsamp']
            nchans = header['frame_shape'][0]
            if self.dispersion_measure is None:
                try:
                    self.dispersion_measure = header[
                        'dispersion_measure']
                except:
                    self.dispersion_measure = 0
            for span in sequence.read(self.gulp_size):
                array_size = span.data.shape[1]/nchans
                frequency = header['fch1']
                for chan in range(nchans):
                    modified_tstart = \
                        tstart - self.calculate_delay(
                            frequency, header['fch1'])
                    ## Sort the data according to which bin
                    ## it should be placed in
                    sort_indices = np.argsort(
                        self.calculate_bin_indices(
                            modified_tstart, tsamp, 
                            array_size, period, bins))
                    sorted_data = span.data[0][chan::nchans][sort_indices]
                    ## So that we can reshape the data before
                    ## summing, disperse zeros throughout the
                    ## data so the size is an integer multiple of
                    ## bins
                    extra_elements = np.round(bins*(1-np.modf(
                        float(array_size)/bins)[0])).astype(int)
                    insert_index = np.floor(
                        np.arange(
                            extra_elements,
                            step=1.0)*float(array_size)/extra_elements)
                    sorted_data = np.insert(
                        sorted_data, insert_index,
                        np.zeros(extra_elements))
                    ## Sum the data into our histogram
                    self.output_array += np.sum(
                        sorted_data.reshape(100, -1), 1)
                    frequency -= header['foff']
                tstart += tsamp*self.gulp_size

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

