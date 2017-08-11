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

"""Test all aspects of `bifrost.blocks.accumulate`"""

import unittest
import bifrost as bf

import bifrost.pipeline as bfp
import bifrost.blocks as blocks


class CallbackBlock(blocks.CopyBlock):
    """Testing-only block which calls user-defined
        functions on sequence and on data"""
    def __init__(self, iring, seq_callback, data_callback, *args, **kwargs):
        super(CallbackBlock, self).__init__(iring, *args, **kwargs)
        self.seq_callback  = seq_callback
        self.data_callback = data_callback
    def on_sequence(self, iseq):
        self.seq_callback(iseq)
        return super(CallbackBlock, self).on_sequence(iseq)
    def on_data(self, ispan, ospan):
        self.data_callback(ispan, ospan)
        return super(CallbackBlock, self).on_data(ispan, ospan)

class TestAccumulateBlock(unittest.TestCase):
    def setUp(self):
        """Create settings shared between tests"""
        self.fil_file = "./data/2chan4bitNoDM.fil"
        self.gulp_nframe = 101
        self.shape_settings = [-1, 1, 2]
    def check_sequence_before(self, seq):
        """Function passed to `CallbackBlock`, which
            checks sequence before accumulate"""
        tensor = seq.header['_tensor']
        self.assertEqual(tensor['shape'], [-1, 1, 2])
        self.assertEqual(tensor['dtype'], 'u8')
        self.assertEqual(tensor['labels'], ['time', 'pol', 'freq'])
        self.assertEqual(tensor['units'], ['s', None, 'MHz'])
    def check_data_before(self, ispan, ospan):
        """Function passed to `CallbackBlock`, which
            checks data before accumulate"""
        self.assertLessEqual(ispan.nframe, self.gulp_nframe)
        self.assertEqual(ospan.nframe, ispan.nframe)
        self.assertEqual(ispan.data.shape, (ispan.nframe, 1, 2))
        self.assertEqual(ospan.data.shape, (ospan.nframe, 1, 2))
    def check_sequence_after(self, seq):
        """Function passed to `CallbackBlock`, which
            checks sequence after accumulate"""
        tensor = seq.header['_tensor']
        self.assertEqual(tensor['shape'], self.shape_settings)
        self.assertEqual(tensor['dtype'], 'u8')
        self.assertEqual(tensor['labels'], ['time', 'pol', 'freq'])
        self.assertEqual(tensor['units'], ['s', None, 'MHz'])
    def check_data_after(self, ispan, ospan):
        """Function passed to `CallbackBlock`, which
            checks data after accumulate"""
        self.assertLessEqual(ispan.nframe, self.gulp_nframe)
        self.assertEqual(ospan.nframe, ispan.nframe)
        self.assertEqual(ispan.data.shape, (ispan.nframe, 1, 2))
        self.assertEqual(ospan.data.shape, (ospan.nframe, 1, 2))
    def test_null_accumulate(self):
        """Check that accumulating no spans leaves header intact"""
        self.shape_settings = [-1, 1, 2]
        with bfp.Pipeline() as pipeline:
            c_data = blocks.sigproc.read_sigproc([self.fil_file], self.gulp_nframe)
            g_data = blocks.copy(c_data, space='cuda')
            call_data = CallbackBlock(
                    g_data, self.check_sequence_before, self.check_data_before)
            accumulated = blocks.accumulate(g_data, 1)
            call_data = CallbackBlock(
                    accumulated, self.check_sequence_after, self.check_data_after)
            pipeline.run()
