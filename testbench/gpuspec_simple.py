#!/usr/bin/env python

# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
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

import bifrost as bf
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Command line utility for creating"
                            "spectra from GuppiRaw files.")
    parser.add_argument('filenames', nargs='+', type=str,
                        help='Names of files to read')
    parser.add_argument('-f', default=1, dest='f_avg', type=int,
                        help='Number of channels to average together after FFT')
    parser.add_argument('-N', default=1, dest='n_int', type=int,
                        help='number of integrations per dump')
    args = parser.parse_args()

    print "Building pipeline"
    bc = bf.BlockChainer()
    bc.blocks.read_guppi_raw(args.filenames, core=0)
    bc.blocks.copy(space='cuda', core=1)
    with bf.block_scope(fuse=True, gpu=0):
        bc.blocks.transpose(['time', 'pol', 'freq', 'fine_time'])
        bc.blocks.fft(axes='fine_time', axis_labels='fine_freq', apply_fftshift=True)
        bc.blocks.detect('stokes')
        bc.views.merge_axes('freq', 'fine_freq')
        bc.blocks.reduce('freq', args.f_avg)
        bc.blocks.accumulate(args.n_int)
    bc.blocks.copy(space='cuda_host', core=2)
    bc.blocks.write_sigproc(core=3)
    print "Running pipeline"
    bf.get_default_pipeline().shutdown_on_signals()
    bf.get_default_pipeline().run()
    print "All done"
