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

class DedisperseBlock(object):
    """This block performs a dedispersion on sigproc-formatted
        data"""
    def __init__(self, input_ring, output_ring):
        self.input_ring = input_ring
        self.output_ring = output_ring
    def dedisperse(self, dispersion_measure):
        """Dedisperse on input ring, moving to output ring.
        @param[in] dispersion_measure Specify the dispersion
            measure that we will remove in the data"""
        pass
    def main(self):
        """Initiate the block's processing"""
        self.dedisperse(dispersion_measure=0)

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
        pass
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
    filenames = ['/data1/mcranmer/data/fake/pulsar.fil']
    blocks = []
    blocks.append(SigprocReadBlock(filenames, raw_data_ring))
    #TODO: Put in a block for RFI cleaning
    blocks.append(DedisperseBlock(raw_data_ring, cleaned_data_ring))
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
    print "Done"

if __name__ == "__main__":
    build_pipeline()
