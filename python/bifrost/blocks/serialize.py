
# Copyright (c) 2016-2020, The Bifrost Authors. All rights reserved.
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

# Python2 compatibility
from __future__ import absolute_import, print_function
import sys
if sys.version_info < (3,):
    range = xrange
    
from bifrost.pipeline import SinkBlock, SourceBlock
import os
try:
    import simplejson as json
except ImportError:
    print("WARNING: Install simplejson for better performance")
    import json
import glob
from functools import reduce

def _parse_bifrost_filename(fname):
    inds = fname[fname.find('.bf.') + 4:].split('.')[:-1]
    inds = [int(i) for i in inds]
    frame0, ringlet_inds = inds[0], inds[1:]
    return frame0, ringlet_inds

class BifrostReader(object):
    def __init__(self, basename):
        assert(basename.endswith('.bf'))
        hdr_filename = basename + '.json'
        with open(hdr_filename, 'r') as hdr_file:
            self.header = json.load(hdr_file)
        data_filenames = glob.glob(basename + '.*.dat')
        inds = [_parse_bifrost_filename(fname) for fname in data_filenames]
        frame0s, ringlet_inds = zip(*inds)
        nringlets = [max(r) + 1 for r in zip(*ringlet_inds)]
        # TODO: Support multiple ringlet axes (needed in SerializeBlock too)
        assert(len(nringlets) <= 1)
        self.nringlet = nringlets[0] if len(nringlets) else 0
        if self.nringlet > 0:
            ringlet_inds = [inds[0] for inds in ringlet_inds]
            self.ringlet_files = []
            for ringlet in range(self.nringlet):
                ringlet_filenames = [f for f, r in zip(data_filenames, ringlet_inds)
                                     if r == ringlet]
                ringlet_filenames.sort()
                ringlet_files = [open(f, 'rb') for f in ringlet_filenames]
                self.ringlet_files.append(ringlet_files)
            self.nfile = len(self.ringlet_files[0])
            if not all([len(files) == self.nfile for files in self.ringlet_files]):
                raise IOError("Number of files in each ringlet does not match")
        else:
            data_filenames.sort()
            self.files = [open(f, 'rb') for f in data_filenames]
            self.nfile = len(self.files)
        self.cur_file = 0
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        if self.nringlet > 0:
            for ringlet in self.ringlet_files:
                for f in ringlet:
                    f.close()
        else:
            for f in self.files:
                f.close()
    def readinto(self, buf, frame_nbyte):
        if self.cur_file == self.nfile:
            return 0
        nframe_read = 0
        if self.nringlet > 0:
            # First dimension of buf is ringlets
            bufs = buf
            nbyte_reads = [ringlet_file[self.cur_file].readinto(buf)
                           for ringlet_file, buf in zip(self.ringlet_files, bufs)]
            nbyte_read = min(nbyte_reads)
        else:
            nbyte_read = self.files[self.cur_file].readinto(buf)
        if nbyte_read % frame_nbyte != 0:
            raise IOError("Unexpected end of file")
        nframe_read += nbyte_read // frame_nbyte
        while nbyte_read < buf.nbytes:
            self.cur_file += 1
            if self.cur_file == self.nfile:
                break
            if self.nringlet > 0:
                nbyte_reads = [ringlet_file[self.cur_file].readinto(buf)
                               for ringlet_file, buf in zip(self.ringlet_files, bufs)]
                nbyte_read = min(nbyte_reads)
            else:
                nbyte_read = self.files[self.cur_file].readinto(buf)
            if nbyte_read % frame_nbyte != 0:
                raise IOError("Unexpected end of file")
            nframe_read += nbyte_read // frame_nbyte
        return nframe_read

class DeserializeBlock(SourceBlock):
    def __init__(self, filenames, gulp_nframe, *args, **kwargs):
        super(DeserializeBlock, self).__init__(filenames, gulp_nframe, *args, **kwargs)
    def create_reader(self, sourcename):
        return BifrostReader(sourcename)
    def on_sequence(self, ireader, sourcename):
        hdr = ireader.header
        return [ireader.header]
    def on_data(self, reader, ospans):
        ospan = ospans[0]
        return [reader.readinto(ospan.data, ospan.frame_nbyte)]

def deserialize(filenames, gulp_nframe, *args, **kwargs):
    """Deserializes a data stream from a set of files using a simple data format

    Sequence headers are read as JSON files, and sequence data are read
    directly as binary from separate files.

    The actual header and data files must have the following general form::

        # Header
        <filename>.json

        # Single-ringlet data
        <filename>.<frame_offset>.dat

        # Multi-ringlet data
        <filename>.<frame_offset>.<ringlet>.dat

    See also: ``serialize``

    Args:
        filenames (list): List of input filenames (each ending with '.bf')
        gulp_nframe (int): No. frames to read at a time.
        *args: Arguments to ``bifrost.pipeline.SourceBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.SourceBlock``.

    **Tensor semantics**::

        Input:  One data file per sequence
        Output: [frame, ...], dtype = any, space = SYSTEM

        Input:  One data file per ringlet
        Output: [ringlet, frame, ...], dtype = any, space = SYSTEM

    Returns:
        DeserializeBlock: A new block instance.
    """
    return DeserializeBlock(filenames, gulp_nframe, *args, **kwargs)

# **TODO: Write a DeserializeBlock that does the inverse of this
class SerializeBlock(SinkBlock):
    def __init__(self, iring, path, max_file_size=None, *args, **kwargs):
        super(SerializeBlock, self).__init__(iring, *args, **kwargs)
        if path is None:
            path = ''
        self.path = path
        if max_file_size is None:
            max_file_size = 1024**3
        self.max_file_size = max_file_size
    def _close_data_files(self):
        if hasattr(self, 'ofiles'):
            for ofile in self.ofiles:
                ofile.close()
    def _open_new_data_files(self, frame_offset):
        self._close_data_files()
        self.bytes_written = 0
        if self.frame_axis == 0:
            # No ringlets, we can write all data to one file
            filenames = [self.basename + '.bf.%012i.dat' % frame_offset]
        elif self.frame_axis == 1:
            # Ringlets, we must write each to a separate file
            ndigit    = len(str(self.nringlet-1))
            filenames = [self.basename + ('.bf.%012i.%0'+str(ndigit)+'i.dat') %
                         (frame_offset, i)
                         for i in range(self.nringlet)]
        else:
            # TODO: Need to deal with separating multiple ringlet axes
            #         E.g., separate each ringlet dim with a dot
            #         Will have to lift/project the indices
            raise NotImplementedError("Multiple ringlet axes not supported")
        # Open data files
        self.ofiles = [open(fname, 'wb') for fname in filenames]
    def on_sequence(self, iseq):
        hdr = iseq.header
        tensor = hdr['_tensor']
        if hdr['name'] != '':
            self.basename = hdr['name']
        else:
            self.basename = '%020i' % hdr['time_tag']
        if self.path != '':
            # TODO: May need more flexibility in path handling
            #         E.g., may want to keep subdirs from original name
            self.basename = os.path.basename(self.basename)
            self.basename = os.path.join(self.path, self.basename)
        # Write sequence header file
        with open(self.basename + '.bf.json', 'w') as hdr_file:
            hdr_file.write(json.dumps(hdr, indent=4, sort_keys=True))
        shape = tensor['shape']
        self.frame_axis = shape.index(-1)
        self.nringlet = reduce(lambda a, b: a * b, shape[:self.frame_axis], 1)
        self._open_new_data_files(frame_offset=0)
    def on_sequence_end(self, iseq):
        self._close_data_files()
    def on_data(self, ispan):
        if self.nringlet == 1:
            bytes_to_write = ispan.data.nbytes
        else:
            bytes_to_write = ispan.data[0].nbytes
        # Check if file size limit has been reached
        if self.bytes_written + bytes_to_write > self.max_file_size:
            self._open_new_data_files(ispan.frame_offset)
        self.bytes_written += bytes_to_write
        # Write data to file(s)
        if self.nringlet == 1:
            ispan.data.tofile(self.ofiles[0])
        else:
            for r in range(self.nringlet):
                ispan.data[r].tofile(self.ofiles[r])

def serialize(iring, path=None, max_file_size=None, *args, **kwargs):
    """Serializes a data stream to a set of files using a simple data format

    Sequence headers are written as JSON files, and sequence data are written
    directly as binary to separate files.

    Filenames begin with the sequence name if present, or the time tag if not.
    The general form is::

        # Header
        <name_or_time_tag>.bf.json

        # Single-ringlet data
        <name_or_time_tag>.bf.<frame_offset>.dat

        # Multi-ringlet data
        <name_or_time_tag>.bf.<frame_offset>.<ringlet>.dat

    Args:
        iring (Ring or Block): Input data source.o
        path (str): Path specifying where to write output files.
        max_file_size (int): Max no. bytes to write to a single file. If set to
            -1, no limit is applied.
        *args: Arguments to ``bifrost.pipeline.SinkBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.SinkBlock``.

    **Tensor semantics**::

        Input:  [frame, ...], dtype = any, space = SYSTEM
        Output: One data file per sequence

        Input:  [ringlet, frame, ...], dtype = any, space = SYSTEM
        Output: One data file per ringlet

    Returns:
        SerializeBlock: A new block instance.
    """
    return SerializeBlock(iring, path, max_file_size, *args, **kwargs)

