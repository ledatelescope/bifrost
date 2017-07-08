
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

import unittest
import bifrost as bf

from bifrost.blocks import *

class CallbackBlock(CopyBlock):
    """Testing-only block which calls user-defined
        functions on sequence and on data"""
    def __init__(self, iring, seq_callback, data_callback, data_ref=None,
                 *args, **kwargs):
        super(CallbackBlock, self).__init__(iring, *args, **kwargs)
        self.seq_callback  = seq_callback
        self.data_callback = data_callback
        self.data_ref = data_ref
    def on_sequence(self, iseq):
        self.seq_callback(iseq)
        return super(CallbackBlock, self).on_sequence(iseq)
    def on_data(self, ispan, ospan):
        self.data_callback(ispan, ospan)
        if self.data_ref is not None:
            # Note: This can be used to check data from outside the pipeline,
            #         which is useful when exceptions inside blocks prevent
            #         downstream callback blocks from ever executing.
            self.data_ref['odata'] = ospan.data.copy()
        return super(CallbackBlock, self).on_data(ispan, ospan)

def identity_block(block, *args, **kwargs):
    return block

def rename_sequence(hdr, name):
    hdr['name'] = name
    return hdr

class PipelineTestCPU(unittest.TestCase):
    def setUp(self):
        # Note: This file needs to be large enough to fill the minimum-size
        #         ring buffer at least a few times over in order to properly
        #         test things.
        self.fil_file = "./data/2chan16bitNoDM.fil"
    def test_read_sigproc(self):
        gulp_nframe = 101
        def check_sequence(seq):
            tensor = seq.header['_tensor']
            self.assertEqual(tensor['shape'],  [-1,1,2])
            self.assertEqual(tensor['dtype'],  'u16')
            self.assertEqual(tensor['labels'], ['time', 'pol', 'freq'])
            self.assertEqual(tensor['units'],  ['s', None, 'MHz'])
        def check_data(ispan, ospan):
            self.assertLessEqual(ispan.nframe, gulp_nframe)
            self.assertEqual(    ospan.nframe, ispan.nframe)
            self.assertEqual(ispan.data.shape, (ispan.nframe,1,2))
            self.assertEqual(ospan.data.shape, (ospan.nframe,1,2))
        with bf.Pipeline() as pipeline:
            data = read_sigproc([self.fil_file], gulp_nframe)
            data = CallbackBlock(data, check_sequence, check_data)
            pipeline.run()
    def run_test_simple_copy(self, guarantee, test_views=False):
        def check_sequence(seq):
            pass
        def check_data(ispan, ospan):
            pass
        gulp_nframe = 101
        with bf.Pipeline() as pipeline:
            data = read_sigproc([self.fil_file], gulp_nframe)
            if test_views:
                data = bf.views.split_axis(data, 'freq', 2, 'fine_freq')
                data = bf.views.rename_axis(data, 'freq', 'chan')
                data = bf.views.merge_axes(data, 'chan', 'fine_freq')
                data = bf.views.astype(data, 'u16')
                data = bf.views.custom(
                    data, lambda hdr: rename_sequence(hdr, hdr['name']))
            for _ in xrange(20):
                data = copy(data, guarantee=guarantee)
            ref = {}
            data = CallbackBlock(data, check_sequence, check_data, data_ref=ref)
            pipeline.run()
            self.assertEqual(ref['odata'].dtype, 'uint16')
            self.assertEqual(ref['odata'].shape, (29, 1, 2))
    def test_simple_copy(self):
        self.run_test_simple_copy(guarantee=True)
    def test_simple_copy_unguaranteed(self):
        self.run_test_simple_copy(guarantee=False)
    def test_simple_views(self):
        self.run_test_simple_copy(guarantee=True, test_views=True)
    def test_simple_views_unguaranteed(self):
        self.run_test_simple_copy(guarantee=False, test_views=True)
    def test_block_chainer(self):
        with bf.Pipeline() as pipeline:
            bc = bf.BlockChainer()
            bc.blocks.read_sigproc([self.fil_file], gulp_nframe=100)
            bc.blocks.transpose(['freq', 'time', 'pol'])
            bc.views.split_axis('time', 1)
            bc.custom(identity_block)()
            pipeline.run()
