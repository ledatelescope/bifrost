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
import os
import threading
import bifrost
import matplotlib
import bandfiles
import numpy as np
from bifrost import affinity
from bifrost.ring import Ring
from bifrost.sigproc import SigprocFile
## Use a graphical backend which supports threading
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
from matplotlib import pylab
from bifrost.block import *

def dada_rficlean_dedisperse_fold_pipeline():
    """This function creates the example pipeline,
        and executes it. It prints 'done' when the execution
        has finished."""
    data_ring = Ring()
    clean_ring = Ring()
    histogram = np.zeros(100).astype(np.float)
    datafilename = '/data1/mcranmer/data/real/2016_xaa.dada'
    imagename = '/data1/mcranmer/data/fake/test_picture.png'
    blocks = []
    blocks.append(
        DadaReadBlock(datafilename, data_ring, gulp_nframe=128))
    #blocks.append(
        #KurtosisBlock(data_ring, clean_ring))
    blocks.append(WaterfallBlock(data_ring, imagename, gulp_nframe=128))
    threads = [threading.Thread(target=block.main) for block in blocks]
    print "Loaded threads"
    for thread in threads:
        thread.daemon = True
        print "Starting thread", thread
        thread.start()
    for thread in threads:
        # wait for thread to terminate
        thread.join()
    # test file has large signal to noise ratio
    print "Done waterfall."


def read_dedisperse_waterfall_pipeline():
    """This function creates the example pipeline,
        and executes it. It prints 'done' when the execution
        has finished."""
    data_ring = Ring()
    histogram = np.zeros(100).astype(np.float)
    datafilename = ['/data1/mcranmer/data/fake/pulsar_DM1_256chan.fil']
    imagename = '/data1/mcranmer/data/fake/test_picture.png'
    blocks = []
    blocks.append(
        SigprocReadBlock(datafilename, data_ring, gulp_nframe=128))
    blocks.append(WaterfallBlock(data_ring, imagename, gulp_nframe=128))
    blocks.append(
        FoldBlock(data_ring, histogram, gulp_size=4096*100, dispersion_measure=1))
    threads = [threading.Thread(target=block.main) for block in blocks]
    print "Loaded threads"
    for thread in threads:
        thread.daemon = True
        print "Starting thread", thread
        thread.start()
    for thread in threads:
        # wait for thread to terminate
        thread.join()
    # test file has large signal to noise ratio
    print "Done waterfall."
    print histogram

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
    assert np.max(histogram)/np.min(histogram) > 10


def read_dedisperse_and_fold_pipeline():
    """This function creates the example pipeline,
        and executes it. It prints 'done' when the execution
        has finished."""
    data_ring = Ring()
    histogram = np.zeros(100).astype(np.float)
    filename = ['/data1/mcranmer/data/fake/simple_pulsar_DM10_128ch.fil']
    blocks = []
    blocks.append(SigprocReadBlock(filename, data_ring))
    blocks.append(DedisperseBlock(data_ring))
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
    assert np.max(histogram)/np.min(histogram) > 10


if __name__ == "__main__":
    dada_rficlean_dedisperse_fold_pipeline()
    #read_dedisperse_waterfall_pipeline()
    #read_and_fold_pipeline()
    #read_and_fold_pipeline_128chan()
    #read_dedisperse_and_fold_pipeline()
