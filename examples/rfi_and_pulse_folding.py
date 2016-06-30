"""@package rfi_and_pulse_folding
This module is example code for creating a pipeline that 
reads data containing a pulsar signal, then removes RFI
in that signal, and then dedisperses and folds the pulse.
"""
import json
import threading
import bifrost
import time
import numpy as np
from matplotlib import pyplot as plt
from bifrost import affinity
from bifrost.ring import Ring
from bifrost.sigproc import SigprocFile

class SigprocReadBlock(object):
    def __init__(self, filenames, outring, gulp_nframe=4096, core=-1):
        self.filenames   = filenames
        self.oring       = outring
        self.gulp_nframe = gulp_nframe
        self.core        = core
    def main(self): # Launched in thread
        affinity.set_core(self.core)
        with self.oring.begin_writing() as oring:
            for name in self.filenames:
                print "Opening", name
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
                    print 'ohdr:', ohdr
                    ohdr = json.dumps(ohdr)
                    gulp_nbyte = self.gulp_nframe*ifile.nchans*ifile.nifs*ifile.nbits/8
                    self.oring.resize(gulp_nbyte)
                    with oring.begin_sequence(name, header=ohdr) as osequence:
                        while True:
                            with osequence.reserve(gulp_nbyte) as wspan:
                                size = ifile.file_object.readinto(wspan.data.data)
                                wspan.commit(size)
                                #print wspan.data.shape
                                #print size
                                if size == 0:
                                    break

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
        self.input_ring = input_ring
        self.output_ring = output_ring
        self.core = core
    def main(self):
        """Initiate the block's processing"""
        affinity.set_core(self.core)
        self.rfi_clean()
    def rfi_clean(self):
        """Calls a kurtosis algorithm and uses the result
            to clean the input data of RFI, and move it to the
            output ring."""
        duplicate_ring(self.input_ring, self.output_ring)

class DedisperseBlock(object):
    """This block performs a dedispersion on sigproc-formatted
        data in a ring"""
    def __init__(self, input_ring, output_ring, core=-1):
        """
        @param[in] input_ring Ring containing a 1d
            timeseries
        @param[out] output_ring Ring will contain a 1d
            timeseries that will be dedispersed
        @param[in] core Which OpenMP core to use for 
            this block. (-1 is any)
        """
        self.input_ring = input_ring
        self.output_ring = output_ring
        self.core = core
    def main(self):
        """Initiate the block's processing"""
        affinity.set_core(self.core)
        self.dedisperse(dispersion_measure=0)
    def dedisperse(self, dispersion_measure):
        """Dedisperse on input ring, moving to output ring.
        @param[in] dispersion_measure Specify the dispersion
            measure that we will remove in the data"""
        duplicate_ring(self.input_ring, self.output_ring)

class FoldBlock(object):
    """This block folds a signal into a histogram"""
    def __init__(self, input_ring, np_output_array, core=-1):
        """
        @param[in] input_ring Ring containing a 1d
            timeseries
        @param[out] np_output_array Numpy array which will
            eventually contain a histogram from our folding
        @param[in] core Which OpenMP core to use for 
            this block. (-1 is any)
        """
        self.input_ring = input_ring
        self.output_array = np_output_array
        self.core = core
    def main(self):
        """Initiate the block's processing"""
        affinity.set_core(self.core)
        self.fold(period=1e-3, bins=100)
    def fold(self, period, bins):
        """Fold the signal into the numpy array
        @param[in] period Period to fold over in seconds
        @param[in] bins Number of bins in the histogram
        """
        gulp_size = 4096
        self.input_ring.resize(gulp_size)
        for sequence in self.input_ring.read(guarantee=True):
            header_ascii = "".join(
                [chr(item) for item in sequence.header])
            header = json.loads(header_ascii)
            tstart = header['tstart']
            tsamp = header['tsamp']
            for span in sequence.read(gulp_size):
                array_size = span.data.shape[1]
                ## Calculate the bin that each data element should
                ## be added to
                arrival_time = tstart+tsamp*np.arange(array_size)
                phase = np.fmod(arrival_time, period)
                bin_index = np.floor(phase/period*bins).astype(int)
                ## Sort the data according to these bins
                sort_indices = np.argsort(bin_index)
                sorted_data = span.data[0][sort_indices[::-1]]
                ## So that we can reshape the data before summing,
                ## disperse zeros throughout the data show the size
                ## is an integer multiple of bins
                extra_elements = np.round(bins*(1-np.modf(
                    float(array_size)/bins)[0]))
                insert_index = np.floor(
                    np.arange(
                        extra_elements,
                        step=1.0)*float(array_size)/extra_elements)
                sorted_data = np.insert(
                    sorted_data, insert_index,
                    np.zeros(extra_elements))
                ## Sum the data into the histogram
                self.output_array += np.sum(
                    sorted_data.reshape(100, -1), 1)
                tstart += tsamp*gulp_size

def build_pipeline():
    """This function creates the example pipeline,
        and executes it. It prints 'done' when the execution
        has finished."""
    raw_data_ring = Ring()
    cleaned_data_ring = Ring()
    dedispersed_data_ring = Ring()
    histogram = np.zeros(100).astype(np.float)
    filenames = ['/data1/mcranmer/data/fake/simple_pulsar_DM0.fil']
    blocks = []
    blocks.append(SigprocReadBlock(filenames, raw_data_ring))
    blocks.append(KurtosisBlock(raw_data_ring, cleaned_data_ring))
    blocks.append(DedisperseBlock(cleaned_data_ring, dedispersed_data_ring))
    blocks.append(FoldBlock(dedispersed_data_ring, histogram))
    threads = [threading.Thread(target=block.main) for block in blocks]

    for thread in threads:
        thread.daemon = True
        thread.start()
    for thread in threads:
        # wait for thread to terminate
        thread.join()

if __name__ == "__main__":
    build_pipeline()
