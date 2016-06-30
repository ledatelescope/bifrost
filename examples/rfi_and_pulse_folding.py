# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of The Bifrost Authors nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""@package rfi_and_pulse_folding
This module is example code for creating a pipeline that
reads data containing a pulsar signal, then removes RFI
in that signal, and then dedisperses and folds the pulse.
"""
import json
import threading
import bifrost
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
                    ohdr['fch1'] = float(ifile.header['fch1'])
                    ohdr['foff'] = float(ifile.header['foff'])
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
    """This block calculates the dedispersion of 
        sigproc-formatted data in a ring, and tags
        it in the headers"""
    def __init__(
            self, ring, core=-1,
            gulp_size=4096):
        """
        @param[in] input_ring Ring containing a 1d
            timeseries
        @param[out] output_ring Ring will contain a 1d
            timeseries that will be dedispersed
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
                        float(array_size)/bins)[0]))
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

def read_and_fold_pipeline():
    """This function creates a pipeline that reads
        in a sigproc file and executes it. 
        It prints 'done' when the execution has finished."""
    raw_data_ring = Ring()
    histogram = np.zeros(100).astype(np.float)
    filenames = ['/data1/mcranmer/data/fake/simple_pulsar_DM0.fil']
    blocks = []
    blocks.append(SigprocReadBlock(filenames, raw_data_ring))
    blocks.append(FoldBlock(raw_data_ring, histogram))
    threads = [threading.Thread(target=block.main) for block in blocks]

    for thread in threads:
        thread.daemon = True
        thread.start()
    for thread in threads:
        # wait for thread to terminate
        thread.join()
    ## Make sure that the histogram is not flat
    assert (np.max(histogram)/np.min(histogram) > 3)

def read_and_fold_pipeline_128chan():
    """This function creates a pipeline that reads
        in a sigproc file and executes it. 
        It prints 'done' when the execution has finished."""
    raw_data_ring = Ring()
    histogram = np.zeros(100).astype(np.float)
    filenames = ['/data1/mcranmer/data/fake/simple_pulsar_DM0_128ch.fil']
    blocks = []
    blocks.append(SigprocReadBlock(filenames, raw_data_ring))
    blocks.append(
        FoldBlock(raw_data_ring, histogram, gulp_size=4096*100))
    threads = [threading.Thread(target=block.main) for block in blocks]

    for thread in threads:
        thread.daemon = True
        thread.start()
    for thread in threads:
        # wait for thread to terminate
        thread.join()
    ## Make sure that the histogram is not flat
    assert (np.max(histogram)/np.min(histogram) > 3)

def read_dedisperse_and_fold_pipeline():
    """This function creates the example pipeline,
        and executes it. It prints 'done' when the execution
        has finished."""
    data_ring = Ring()
    histogram = np.zeros(100).astype(np.float)
    histogram_no_dm = np.zeros(100).astype(np.float)
    filenames = ['/data1/mcranmer/data/fake/simple_pulsar_DM10_128ch.fil']
    blocks = []
    blocks.append(SigprocReadBlock(filenames, data_ring))
    blocks.append(DedisperseBlock(data_ring))
    blocks.append(FoldBlock(
        data_ring, histogram_no_dm, 
        gulp_size=4096*128*100, 
        dispersion_measure=0))
    blocks.append(FoldBlock(
        data_ring, histogram, 
        gulp_size=4096*128*100, 
        dispersion_measure=10))
    threads = [threading.Thread(target=block.main) for block in blocks]

    for thread in threads:
        thread.daemon = True
        thread.start()
    for thread in threads:
        # wait for thread to terminate
        thread.join()
    # test file has large signal to noise ratio
    assert (np.max(histogram)/np.min(histogram) > 10)

if __name__ == "__main__":
    read_and_fold_pipeline()
    read_and_fold_pipeline_128chan()
    read_dedisperse_and_fold_pipeline()
