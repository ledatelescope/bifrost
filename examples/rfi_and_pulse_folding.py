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
import numpy as np
from bifrost.ring import Ring
from bifrost.sigproc import SigprocFile
from bifrost import affinity

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
                    ohdr['frame_shape']   = (ifile.nchans, ifile.nifs)
                    ohdr['frame_size']    = ifile.nchans*ifile.nifs
                    ohdr['frame_nbyte']   = ifile.nchans*ifile.nifs*ifile.nbits/8
                    ohdr['frame_axes']    = ('pol', 'chan')
                    ohdr['ringlet_shape'] = (1,)
                    ohdr['ringlet_axes']  = ()
                    ohdr['dtype']         = str(ifile.dtype)
                    ohdr['nbit']          = ifile.nbits
                    print 'ohdr:', ohdr
                    ohdr = json.dumps(ohdr)
                    gulp_nbyte = self.gulp_nframe*ifile.nchans*ifile.nifs*ifile.nbits/8
                    self.oring.resize(gulp_nbyte)
                    with oring.begin_sequence(name, header=ohdr) as osequence:
                        while True:
                            with osequence.reserve(gulp_nbyte) as wspan:
                                size = ifile.file_object.readinto(wspan.data.data)
                                wspan.commit(size)
                                #print size
                                if size == 0:
                                    break

class KurtosisBlock(object):
    """This block performs spectral kurtosis and cleaning
        on sigproc-formatted data in rings"""
    def __init__(self, input_ring, output_ring):
        """
        @param[in] input_ring Ring containing a 1d
            timeseries
        @param[out] output_ring Ring will contain a 1d
            timeseries that will be cleaned of RFI 
        """
        self.input_ring = input_ring
        self.output_ring = output_ring
    def main(self):
        """Initiate the block's processing"""
        self.rfi_clean()
    def rfi_clean(self):
        """Calls a kurtosis algorithm and uses the result
            to clean the input data of RFI, and move it to the
            output ring."""
        self.output_ring = self.input_ring

class DedisperseBlock(object):
    """This block performs a dedispersion on sigproc-formatted
        data in a ring"""
    def __init__(self, input_ring, output_ring):
        """
        @param[in] input_ring Ring containing a 1d
            timeseries
        @param[out] output_ring Ring will contain a 1d
            timeseries that will be dedispersed
        """
        self.input_ring = input_ring
        self.output_ring = output_ring
    def main(self):
        """Initiate the block's processing"""
        self.dedisperse(dispersion_measure=0)
    def dedisperse(self, dispersion_measure):
        """Dedisperse on input ring, moving to output ring.
        @param[in] dispersion_measure Specify the dispersion
            measure that we will remove in the data"""
        self.output_ring = self.input_ring

class FoldBlock(object):
    """This block folds a signal into a histogram"""
    def __init__(self, input_ring, np_output_buffer):
        """
        @param[in] input_ring Ring containing a 1d
            timeseries
        @param[out] np_output_buffer Numpy array which will
        eventually contain a histogram from our folding
        """
        self.input_ring = input_ring
        self.output_buffer = np_output_buffer
    def fold(self, period, bins):
        """
        @param[in] period Period to fold over in seconds
        @param[in] bins Number of bins in the histogram
        """
        self.output_buffer
    def main(self):
        """Initiate the block's processing"""
        self.fold(period=1.0, bins=100)

def build_pipeline():
    """This function creates the example pipeline,
    and executes it. It prints 'done' when the execution
    has finished."""
    raw_data_ring = Ring()
    cleaned_data_ring = Ring()
    histogram = np.zeros(100).astype(np.float)
    filenames = ['/data1/mcranmer/data/fake/simple_pulsar_DM0.fil']
    blocks = []
    blocks.append(SigprocReadBlock(filenames, raw_data_ring))
    blocks.append(KurtosisBlock(raw_data_ring, cleaned_data_ring))
    blocks.append(DedisperseBlock(cleaned_data_ring, cleaned_data_ring))
    blocks.append(FoldBlock(cleaned_data_ring, histogram))
    threads = [threading.Thread(target=block.main) for block in blocks]
    print "Launching %i threads" % len(threads)
    for thread in threads:
        thread.daemon = True
        thread.start()
    print "Waiting for threads to finish"
    #while not shutdown_event.is_set():
    #	signal.pause()
    for thread in threads:
        thread.join()
    

if __name__ == "__main__":
    build_pipeline()
