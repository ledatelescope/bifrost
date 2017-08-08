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
from bifrost.dtype import name_nbit2numpy

class BinaryFileRead(object):
    """ Simple file-like reading object for pipeline testing

    Args:
        filename (str): Name of file to open
        dtype (np.dtype or str): datatype of data, e.g. float32. This should be a *numpy* dtype,
                                 not a bifrost.ndarray dtype (eg. float32, not f32)
        gulp_size (int): How much data to read per gulp, (i.e. sub-array size)
    """
    def __init__(self, filename, gulp_size, dtype):
        super(BinaryFileRead, self).__init__()
        self.file_obj = open(filename, 'r')
        self.dtype = dtype
        self.gulp_size = gulp_size

    def read(self):
        d = np.fromfile(self.file_obj, dtype=self.dtype, count=self.gulp_size)
        return d

    def __enter__(self):
        return self

    def close(self):
        pass

    def __exit__(self, type, value, tb):
        self.close()


class BinaryFileReadBlock(bfp.SourceBlock):
    def __init__(self, filenames, gulp_size, gulp_nframe, dtype, *args, **kwargs):
        super(BinaryFileReadBlock, self).__init__(filenames, gulp_nframe, *args, **kwargs)
        self.dtype = dtype
        self.gulp_size = gulp_size

    def create_reader(self, filename):
        # Do a lookup on bifrost datatype to numpy datatype
        dcode = self.dtype.rstrip('0123456789')
        nbits = int(self.dtype[len(dcode):])
        np_dtype = name_nbit2numpy(dcode, nbits)

        return BinaryFileRead(filename, self.gulp_size, np_dtype)

    def on_sequence(self, ireader, filename):
        ohdr = {
            'name': filename,
            '_tensor': {
                'dtype':  self.dtype,
                'shape':  [-1, self.gulp_size],
                'labels':  ['streamed', 'gulped'],
                'units': [None, None],
                'scales': [[0, 1], [0, 1]]
            },
        }
        return [ohdr]

    def on_data(self, reader, ospans):
        indata = reader.read()

        if indata.shape[0] == self.gulp_size:
            ospans[0].data[0] = indata
            return [1]
        else:
            return [0]

class BinaryFileWriteBlock(bfp.SinkBlock):
    def __init__(self, iring, file_ext='out', *args, **kwargs):
        super(BinaryFileWriteBlock, self).__init__(iring, *args, **kwargs)
        self.current_fileobj = None
        self.file_ext = file_ext

    def on_sequence(self, iseq):
        if self.current_fileobj is not None:
            self.current_fileobj.close()

        new_filename = iseq.header['name'] + '.' + self.file_ext
        self.current_fileobj = open(new_filename, 'w')

    def on_data(self, ispan):
        self.current_fileobj.write(ispan.data.tobytes())

def binary_read(filenames, gulp_size, gulp_nframe, dtype, *args, **kwargs):
    """ Block for reading binary data from file and streaming it into a bifrost pipeline

    Args:
        filenames (list): A list of filenames to open
        gulp_size (int): Number of elements in a gulp (i.e. sub-array size)
        gulp_nframe (int): Number of frames in a gulp. (Ask Ben / Miles for good explanation)
        dtype (bifrost dtype string): dtype, e.g. f32, cf32
    """
    return BinaryFileReadBlock(filenames, gulp_size, gulp_nframe, dtype, *args, **kwargs)

def binary_write(iring, file_ext='out', *args, **kwargs):
    """ Write ring data to a binary file

    Args:
        file_ext (str): Output file extension. Defaults to '.out'

    Notes:
        output filename is generated from the header 'name' keyword + file_ext
    """
    return BinaryFileWriteBlock(iring, file_ext, *args, **kwargs)
