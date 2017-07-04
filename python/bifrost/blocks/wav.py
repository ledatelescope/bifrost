
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

from __future__ import absolute_import

from bifrost.pipeline import SourceBlock, SinkBlock
from bifrost.DataType import DataType
from bifrost.units import convert_units

import struct
import os

def wav_read_chunk_desc(f):
    id_, size, fmt = struct.unpack('<4sI4s', f.read(12))
    return id_, size, fmt
def wav_read_subchunk_desc(f):
    id_, size = struct.unpack('<4sI', f.read(8))
    return id_, size
def wav_read_subchunk_fmt(f, size):
    assert(size >= 16)
    packed = f.read(16)
    f.seek(size - 16, 1)
    keys = ('audio_fmt', 'nchan', 'sample_rate', 'byte_rate',
            'block_align', 'nbit')
    vals = struct.unpack('<HHIIHH', packed)
    info = {k: v for k, v in zip(keys, vals)}
    return info
def wav_read_header(f):
    # **TODO: Some files actually have extra subchunks _after_ the data as well
    #           This is rather annoying :/
    chunk_id, chunk_size, chunk_fmt = wav_read_chunk_desc(f)
    assert(chunk_id  == 'RIFF')
    assert(chunk_fmt == 'WAVE')
    hdr = None
    subchunk_id, subchunk_size = wav_read_subchunk_desc(f)
    while subchunk_id != 'data':
        if subchunk_id == 'fmt ':
            hdr = wav_read_subchunk_fmt(f, subchunk_size)
        else:
            f.seek(subchunk_size, 1) # Ignore any other subchunks
        subchunk_id, subchunk_size = wav_read_subchunk_desc(f)
    data_size = subchunk_size
    return hdr, data_size
def wav_write_header(f, hdr, chunk_size=0, data_size=0):
    # Note: chunk_size = file size - 8
    f.write(struct.pack('<4sI4s4sIHHIIHH4sI',
                        'RIFF', chunk_size, 'WAVE',
                        'fmt ', 16,
                        hdr['audio_fmt'], hdr['nchan'], hdr['sample_rate'],
                        hdr['byte_rate'], hdr['block_align'], hdr['nbit'],
                        'data', data_size))

class WavSourceBlock(SourceBlock):
    def create_reader(self, sourcename):
        return open(sourcename, 'rb')
    def on_sequence(self, reader, sourcename):
        hdr, self.bytes_remaining = wav_read_header(reader)
        ohdr = {
            '_tensor': {
                'dtype':  'u8' if hdr['nbit'] == 8 else 'i' + str(hdr['nbit']),
                'shape':  [-1, hdr['nchan']],
                'labels': ['time', 'pol'],
                'scales': [(0, 1. / hdr['sample_rate']),
                           None],
                'units':  ['s', None]
            },
            'frame_rate':   hdr['sample_rate'],
            'name': sourcename
        }
        return [ohdr]

    def on_data(self, reader, ospans):
        ospan = ospans[0]
        nbyte = reader.readinto(ospan.data)
        if nbyte % ospan.frame_nbyte:
            raise IOError("Input file is truncated")
        # Note: This ensures only the data subchunk is read, avoiding any
        #         subchunks that appear after the data.
        nbyte = min(nbyte, self.bytes_remaining)
        self.bytes_remaining -= nbyte
        # HACK TESTING avoid incomplete final gulp that messes up split_axis
        if nbyte < ospan.data.nbytes:
            return [0]
        nframe = nbyte // ospan.frame_nbyte
        return [nframe]

def read_wav(sourcefiles, gulp_nframe, *args, **kwargs):
    """Read Wave files (.wav).

    Args:
        sourcefiles (list): List of input filenames.
        gulp_nframe (int): No. frames to read at a time.
        *args: Arguments to ``bifrost.pipeline.SourceBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.SourceBlock``.

    **Tensor semantics**::

        Output: ['time', 'pol'], dtype = u8 or i*, space = SYSTEM

    Returns:
        WavSourceBlock: A new block instance.

    """
    return WavSourceBlock(sourcefiles, gulp_nframe, *args, **kwargs)

class WavSinkBlock(SinkBlock):
    def __init__(self, iring, path=None, *args, **kwargs):
        super(WavSinkBlock, self).__init__(iring, *args, **kwargs)
        if path is None:
            path = ''
        self.path = path
    def on_sequence(self, iseq):
        ihdr = iseq.header
        itensor = ihdr['_tensor']

        axnames = tuple(itensor['labels'])
        shape   = itensor['shape']
        scales  = itensor['scales']
        units   = itensor['units']
        ndim    = len(shape)
        dtype   = DataType(itensor['dtype'])

        nchan = shape[-1]
        sample_time = convert_units(scales[-2][1], units[-2], 's')
        sample_rate = int(round(1. / sample_time))
        frame_nbyte = nchan * dtype.itemsize
        ohdr = {
            'audio_fmt':   1, # 1 => PCM (linear quantization, uncompressed)
            'nchan':       nchan,
            'sample_rate': sample_rate,
            'byte_rate':   sample_rate * frame_nbyte,
            'block_align': frame_nbyte,
            'nbit':        dtype.itemsize_bits
        }
        filename = os.path.join(self.path, ihdr['name'])

        if ndim == 2 and axnames[-2] == 'time':
            self.ofile = open(filename + '.wav', 'wb')
            wav_write_header(self.ofile, ohdr)
        elif ndim == 3 and axnames[-2] == 'time':
            nfile = shape[-3]
            filenames = [filename + '.%09i.tim' % i for i in xrange(nfile)]
            self.ofiles = [open(fname + '.wav', 'wb') for fname in filenames]
            for ofile in self.ofiles:
                wav_write_header(ofile, ohdr)
        else:
            raise ValueError("Incompatible axes: " + str(axnames))

    def on_sequence_end(self, iseq):
        if hasattr(self, 'ofile'):
            self.ofile.close()
        elif hasattr(self, 'ofiles'):
            for ofile in self.ofiles:
                ofile.close()

    def on_data(self, ispan):
        idata = ispan.data
        if idata.ndim == 2:
            idata.tofile(self.ofile)
        elif idata.ndim == 3:
            for b, ofile in enumerate(self.ofiles):
                idata[b].tofile(ofile)
        else:
            raise ValueError("Internal error: Unknown data format!")

def write_wav(iring, path=None, *args, **kwargs):
    """Write data as Wave files (.wav).

    Args:
        iring (Ring or Block): Input data source.
        path (str): Path specifying where to write output files.
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.

    **Tensor semantics**::

        Input: [time, pol], dtype = u8 or i*, space = SYSTEM
        Output: Wave file, one file per sequence

        Input: [batch, time, pol], dtype = u8 or i*, space = SYSTEM
        Output: Wave file, one file per batch element

    Returns:
        WavSinkBlock: A new block instance.

    Note:
        The chunk_size and data_size entries in the output wav header are
        written as zero values because they are not known a-priori in a
        streaming setting. VLC still plays the files just fine, but any
        subchunks that appear after the data will be misinterpreted as data.
    """
    return WavSinkBlock(iring, path, *args, **kwargs)
