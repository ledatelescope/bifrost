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
# binary_file_io.py

Basic file I/O blocks for reading and writing data.
"""
import numpy as np
import time
import bifrost as bf
import bifrost.pipeline as bfp
from bifrost.dtype import string2numpy
from astropy.time import Time

def _angle_str_to_sigproc(ang):
    aparts = ang.split(':')
    if len(aparts) == 2:
        sp_ang = int(aparts[0])*10000 + int(aparts[1])*100 + 0.0
    elif len(aparts) == 3:
        sp_ang = int(aparts[0])*10000 + int(aparts[1])*100 + float(aparts[2])
    else:
        raise RuntimeError("Cannot parse: {ang} does not match XX:YY:ZZ.zz".format(ang=ang))

class DadaFileRead(object):
    """ Simple file-like reading object for pipeline testing

    Args:
        filename (str): Name of file to open
        dtype (np.dtype or str): datatype of data, e.g. float32. This should be a *numpy* dtype,
                                 not a bifrost.ndarray dtype (eg. float32, not f32)
        gulp_size (int): How much data to read per gulp, (i.e. sub-array size)
    """
    def __init__(self, filename, header_callback):
        super(DadaFileRead, self).__init__()
        if isinstance(filename, str):
            self.file_obj = open(filename, 'rb')
            self.nfiles   = 1
        else:
            self.file_obs = open(filename[0], 'rb')
            self.nfiles   = len(filename)

        self.filenames = filename
        self.fcount = 0

        print("%i/%i: opening %s" % (self.fcount+1, self.nfiles, filename))
        self.header = self._read_header(header_callback)
        itensor = self.header['_tensor']
        self.dtype          = string2numpy(itensor['dtype'])
        self.block_shape    = np.copy(itensor['shape'])
        self.block_shape[0] = 1
        self.block_size = np.prod(self.block_shape)
        self.gulp_size  = self.block_size * self.dtype.itemsize
        
    def _read_header(self, header_callback):
        """ Read DADA header, and apply header_callback 
        
        Applies header_callback to convert DADA header to bifrost header. Specifically,
        need to generate the '_tensor' from the DADA keywords which have no formal spec.
        """
        hdr_buf = self.file_obj.read(4096)
        hdr = {}
        for line in hdr_buf.split('\n'):
            try:
                key, val = line.split()
                hdr[key] = val
            except ValueError:
                pass
        hdr = header_callback(hdr)
        return hdr
    
    def _open_next_file(self):
        self.file_obj.close()
        self.fcount += 1
        print("%i/%i: opening %s" % (self.fcount+1, self.nfiles, self.filenames[self.fcount]))
        self.file_obj = open(self.filenames[self.fcount], 'r')
        _hdr = self._read_header()

    def _read_data(self):
        d = np.fromfile(self.file_obj, dtype=self.dtype, count=self.block_size)
        return d

    def read_block(self):
        #print("Reading...")
        d = self._read_data()
        if d.size == 0 and self.fcount < self.nfiles - 1:
            self._open_next_file()
            d = self._read_data()
            d = d.reshape(self.block_shape)
            return d
        elif d.size == 0 and self.fcount == self.nfiles - 1:    
            #print("EOF")
            return np.array([0]) # EOF
        elif self.block_size != np.prod(d.shape):
            print(self.block_size, np.prod(d.shape), d.shape, d.size)
            print("Warning: File truncated or gulp size does not divide n_blocks")
            try:
                bs_truncated = list(self.block_shape)
                bs_truncated[0] = -1 # Attempt to load truncated data anyway
                d = d.reshape(bs_truncated)
                return d
            except ValueError:
                return np.array([0]) # EOF
        else:
            d = d.reshape(self.block_shape)
            return d
    
    def read(self):
        gulp_break = False
        d = self.read_block()
        return d


    def __enter__(self):
        return self

    def close(self):
        pass

    def __exit__(self, type, value, tb):
        self.close()


class DadaFileReadBlock(bfp.SourceBlock):
    def __init__(self, filename, header_callback, gulp_nframe,  *args, **kwargs):
        super(DadaFileReadBlock, self).__init__(filename, gulp_nframe, *args, **kwargs)
        self.header_callback = header_callback

    def create_reader(self, filename):
        print(filename)
        # Do a lookup on bifrost datatype to numpy datatype
        return DadaFileRead(filename, self.header_callback)

    def on_sequence(self, ireader, filename):
        ohdr = ireader.header
        return [ohdr]

    def on_data(self, reader, ospans):
        indata = reader.read()
        odata  = ospans[0].data
        #print indata.shape, odata.shape
        if np.prod(indata.shape) == np.prod(odata.shape[1:]):
            ospans[0].data[0] = indata
            return [1]
        else:
            # EOF or truncated block
            return [0]


def read_dada_file(filename, header_callback, gulp_nframe, *args, **kwargs):
    """ Block for reading binary data from file and streaming it into a bifrost pipeline

    Args:
        filenames (list): A list of filenames to open
        header_callback (method): A function that converts from PSRDADA header into a 
                                  bifrost header.
        gulp_size (int): Number of elements in a gulp (i.e. sub-array size)
        gulp_nframe (int): Number of frames in a gulp. (Ask Ben / Miles for good explanation)
        dtype (bifrost dtype string): dtype, e.g. f32, cf32
    """
    return DadaFileReadBlock(filename, header_callback, gulp_nframe, *args, **kwargs)

