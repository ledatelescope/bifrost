# Copyright (c) 2019, The Bifrost Authors. All rights reserved.
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
import os
import bifrost as bf
from bifrost.packet_writer import HeaderInfo, DiskWriter
from bifrost.packet_capture import DiskReader
from bifrost.quantize import quantize
import numpy as np


class DiskIOTest(unittest.TestCase):
    """Test simple IO for the disk-based packet reader and writing"""
    def setUp(self):
        """Generate some dummy data to read"""
        # Generate test vector and save to file
        t = np.arange(128*4096*2)
        w = 0.01
        self.s0 = 6*np.cos(w * t, dtype='float32') \
                  + 5j*np.sin(w * t, dtype='float32')

    def test_write_tbn(self):
         desc = HeaderInfo()
         desc.set_chan0(1)
         desc.set_decimation(500)
          
         fh = open('test_tbn.dat', 'wb')
         op = DiskWriter('tbn', fh)
         
         # Reorder as packets, stands, time
         data = self.s0.reshape(512,32,-1)
         data = data.transpose(2,1,0).copy()
         # Convert to ci8 for TBN
	 data_q = bf.ndarray(shape=data.shape, dtype='ci8')
         quantize(data, data_q)

         # Go!
         desc.set_nsrc(data_q.shape[1])
         op.send(desc, 0, 1, 0, 1, data_q)
	 fh.close()
         
         self.assertEqual(os.path.getsize('test_tbn.dat'), \
                          1048*data_q.shape[0]*data_q.shape[1])
         os.unlink('test_tbn.dat')

    def test_write_drx(self):
         desc = HeaderInfo()
         desc.set_chan0(1)
         desc.set_decimation(10)

         fh = open('test_drx.dat', 'wb')
         op = DiskWriter('drx', fh)

         # Reorder as packets, beams, time
         data = self.s0.reshape(4096,4,-1)
         data = data.transpose(2,1,0).copy()
         # Convert to ci4 for DRX
         data_q = bf.ndarray(shape=data.shape, dtype='ci4')
         quantize(data, data_q)

         # Go!
         desc.set_nsrc(data_q.shape[1])
         op.send(desc, 0, 1, 0, 1, data_q)
         fh.close()

         self.assertEqual(os.path.getsize('test_drx.dat'), \
                          4128*data_q.shape[0]*data_q.shape[1])
         os.unlink('test_drx.dat')
