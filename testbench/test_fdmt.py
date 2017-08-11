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

"""
# test_fdmt.py

This file reads a sigproc filterbank file, and applies the Fast Dispersion
Measure Transform (FDMT), writing the output to a PGM file.
"""

import bifrost.pipeline as bfp
from bifrost.blocks import read_sigproc, copy, transpose, fdmt, scrunch
from bifrost import blocks

import os
import numpy as np

# This is a (very hacky) sink block for writing data as a greyscale PGM image
class PgmWriterBlock(bfp.SinkBlock):
    def __init__(self, iring, outpath='', filename_callback=None,
                 *args, **kwargs):
        super(PgmWriterBlock, self).__init__(iring, *args, **kwargs)
        self.outpath = outpath
        self.filename_callback = (filename_callback or
                                  PgmWriterBlock.default_filename_callback)
    @staticmethod
    def default_filename_callback(ihdr):
        return ihdr['name'] + '.pgm'
    def on_sequence(self, iseq):
        """Return: oheaders (one per output) and islices (one per input)
        """
        ihdr = iseq.header
        dtype = ihdr['_tensor']['dtype']
        shape = ihdr['_tensor']['shape']
        maxval = 255
        filename = os.path.join(self.outpath, self.filename_callback(ihdr))
        self.outfile = open(filename, 'wb')
        self.outfile.write("P5\n")
        # HACK This sets the height to gulp_nframe because we don't know the
        #        sequence length apriori.
        self.outfile.write("%i %i\n%i\n" % (shape[-1], ihdr['gulp_nframe'], maxval))
    # **TODO: Need something like on_sequence_end, or a proper SinkBlock class
    def on_data(self, ispan):
        """Process data from from ispans to ospans and return the number of
        frames to commit for each output (or None to commit complete spans)."""
        data = ispan.data
        print "PgmWriterBlock.on_data()"
        # HACK TESTING
        if data.dtype != np.uint8:
            data = (data - data.min()) / (data.max() - data.min()) * 255
            #data = np.clip(data, 0, 255)
            data = data.astype(np.uint8)
            #data = data.astype(np.uint16)
        if self.outfile is None:
            return

        data.tofile(self.outfile)
        # HACK TESTING only write the first gulp
        self.outfile.close()
        self.outfile = None
def write_pgm(iring, *args, **kwargs):
    PgmWriterBlock(iring, *args, **kwargs)

def main():
    import sys
    if len(sys.argv) <= 1:
        print "Usage: example1.py file1.fil [file2.fil ...]"
        sys.exit(-1)
    filenames = sys.argv[1:]

    h_filterbank = read_sigproc(filenames, gulp_nframe=16000, core=0)
    h_filterbank = scrunch(h_filterbank, 16, core=0)
    d_filterbank = copy(h_filterbank, space='cuda', gpu=0, core=2)
    blocks.print_header(d_filterbank)
    with bfp.block_scope(core=2, gpu=0):
        d_filterbankT     = transpose(d_filterbank, ['pol','freq','time'])#[1,2,0])
        d_dispersionbankT = fdmt(d_filterbankT, max_dm=282.52)
        blocks.print_header(d_dispersionbankT)
        d_dispersionbank  = transpose(d_dispersionbankT, ['time','pol','dispersion'])#[2,0,1])
    h_dispersionbank = copy(d_dispersionbank, space='system', core=3)
    write_pgm(h_dispersionbank, core=3)

    pipeline = bfp.get_default_pipeline()
    graph_filename = "example1.dot"
    with open(graph_filename, 'w') as dotfile:
        dotfile.write(str(pipeline.dot_graph()))
        print "Wrote graph definition to", graph_filename
    pipeline.run()
    print "All done"

if __name__ == '__main__':
    main()

