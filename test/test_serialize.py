
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

import unittest
import bifrost as bf

from bifrost.blocks import *

import os
import shutil

class TemporaryDirectory(object):
    def __init__(self, path):
        self.path = path
        os.makedirs(self.path)
    def remove(self):
        shutil.rmtree(self.path)
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.remove()

def get_sigproc_file_size(filename):
    """Returns the header and data size of a sigproc file without reading
    the whole file.
    """
    with open(filename, 'rb') as f:
        head = ''
        while 'HEADER_END' not in head:
            more_data = f.read(4096)
            if len(more_data) == 0:
                raise IOError("Not a valid sigproc file: " + filename)
            head += more_data
        hdr_size = head.find('HEADER_END') + len('HEADER_END')
    file_size = os.path.getsize(filename)
    data_size = file_size - hdr_size
    return hdr_size, data_size

def rename_sequence(hdr, name):
    hdr['name'] = name
    return hdr

class SerializeTest(unittest.TestCase):
    def setUp(self):
        self.fil_file = "./data/2chan16bitNoDM.fil"
        # Note: This is specific to 2chan16bitNoDM.fil
        self.time_tag = 3493024746386227200
        hdr_size, self.data_size = get_sigproc_file_size(self.fil_file)
        with open(self.fil_file, 'rb') as f:
            self.data = f.read()
            self.data = self.data[hdr_size:]
        self.temp_path = '/tmp/bifrost_test_serialize'
        self.basename = os.path.basename(self.fil_file)
        self.basepath = os.path.join(self.temp_path, self.basename)
        self.gulp_nframe = 101
    def test_serialize_with_name_no_ringlets(self):
        with bf.Pipeline() as pipeline:
            data = read_sigproc([self.fil_file], self.gulp_nframe)
            data = serialize(data, self.temp_path)
            with TemporaryDirectory(self.temp_path):
                pipeline.run()
                # Note: SerializeBlock uses os.path.basename if path is given
                hdrpath = self.basepath + '.bf.json'
                datpath = self.basepath + '.' + '0' * 12 + '.bf.dat'
                self.assertTrue(os.path.exists(hdrpath))
                self.assertTrue(os.path.exists(datpath))
                self.assertEqual(os.path.getsize(datpath), self.data_size)
                with open(datpath, 'rb') as f:
                    data = f.read()
                    self.assertEqual(data, self.data)
    def test_serialize_with_time_tag_no_ringlets(self):
        with bf.Pipeline() as pipeline:
            data = read_sigproc([self.fil_file], self.gulp_nframe)
            # Custom view sets sequence name to '', which causes SerializeBlock
            #   to use the time_tag instead.
            data = bf.views.custom(data, lambda hdr: rename_sequence(hdr, ''))
            data = serialize(data, self.temp_path)
            with TemporaryDirectory(self.temp_path):
                pipeline.run()
                basepath = os.path.join(self.temp_path,
                                        '%020i' % self.time_tag)
                hdrpath = basepath + '.bf.json'
                datpath = basepath + '.' + '0' * 12 + '.bf.dat'
                self.assertTrue(os.path.exists(hdrpath))
                self.assertTrue(os.path.exists(datpath))
                self.assertEqual(os.path.getsize(datpath), self.data_size)
                with open(datpath, 'rb') as f:
                    data = f.read()
                    self.assertEqual(data, self.data)
    def test_serialize_with_name_and_ringlets(self):
        with bf.Pipeline() as pipeline:
            data = read_sigproc([self.fil_file], self.gulp_nframe)
            # Transpose so that freq becomes a ringlet dimension
            # TODO: Test multiple ringlet dimensions (e.g., freq + pol) once
            #         SerializeBlock supports it.
            data = transpose(data, ['freq', 'time', 'pol'])
            data = serialize(data, self.temp_path)
            with TemporaryDirectory(self.temp_path):
                pipeline.run()
                # Note: SerializeBlock uses os.path.basename if path is given
                hdrpath  = self.basepath + '.bf.json'
                datpath0 = self.basepath + '.' + '0' * 12 + '.0.bf.dat'
                datpath1 = self.basepath + '.' + '0' * 12 + '.1.bf.dat'
                self.assertTrue(os.path.exists(hdrpath))
                self.assertTrue(os.path.exists(datpath0))
                self.assertTrue(os.path.exists(datpath1))
                self.assertEqual(os.path.getsize(datpath0),
                                 self.data_size // 2)
                self.assertEqual(os.path.getsize(datpath1),
                                 self.data_size // 2)
