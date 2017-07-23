
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

class PipelineTest(unittest.TestCase):
    def setUp(self):
        # Note: This file needs to be large enough to fill the minimum-size
        #         ring buffer at least a few times over in order to properly
        #         test things.
        self.fil_file = "./data/2chan16bitNoDM.fil"
    def test_cuda_copy(self):
        def check_sequence(seq):
            pass
        def check_data(ispan, ospan):
            pass
        gulp_nframe = 101
        with bf.Pipeline() as pipeline:
            data = read_sigproc([self.fil_file], gulp_nframe)
            for _ in xrange(10):
                data = copy(data, space='cuda')
                data = copy(data, space='cuda_host')
            ref = {}
            data = CallbackBlock(data, check_sequence, check_data, data_ref=ref)
            pipeline.run()
            self.assertEqual(ref['odata'].dtype, 'uint16')
            self.assertEqual(ref['odata'].shape, (29, 1, 2))
    def test_fdmt(self):
        gulp_nframe = 101
        # TODO: Check handling of multiple pols (not currently supported?)
        def check_sequence(seq):
            tensor = seq.header['_tensor']
            self.assertEqual(tensor['shape'],  [1,5,-1])
            self.assertEqual(tensor['dtype'],  'f32')
            self.assertEqual(tensor['labels'], ['pol', 'dispersion', 'time'])
            self.assertEqual(tensor['units'],  [None, 'pc cm^-3', 's'])
        def check_data(ispan, ospan):
            # Note: nframe = gulp_nframe + max_delay
            #self.assertLessEqual(ispan.nframe, gulp_nframe)
            self.assertEqual(    ospan.nframe, ispan.nframe)
            self.assertEqual(ispan.data.shape, (1,5,ispan.nframe))
            self.assertEqual(ospan.data.shape, (1,5,ospan.nframe))
        with bf.Pipeline() as pipeline:
            data = read_sigproc([self.fil_file], gulp_nframe)
            data = copy(data, space='cuda')
            data = transpose(data, ['pol', 'freq', 'time'])
            data = fdmt(data, max_dm=30.)
            ref = {}
            data = CallbackBlock(data, check_sequence, check_data, data_ref=ref)
            data = transpose(data, ['time', 'pol', 'dispersion'])
            data = copy(data, space='cuda_host')
            pipeline.run()
            self.assertEqual(ref['odata'].dtype, 'float32')
            self.assertEqual(ref['odata'].shape, (1, 5, 17))
