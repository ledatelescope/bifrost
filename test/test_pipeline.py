
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
import numpy as np
import bifrost as bf

import bifrost.pipeline as bfp
from bifrost.sigproc_block   import read_sigproc
from bifrost.copy_block      import copy, CopyBlock
from bifrost.transpose_block import transpose
from bifrost.fdmt_block      import fdmt

from copy import deepcopy

class CallbackBlock(CopyBlock):
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

class PipelineTest(unittest.TestCase):
	def setUp(self):
		self.fil_file = "./data/2chan4bitNoDM.fil"
	def test_read_sigproc(self):
		gulp_nframe = 101
		def check_sequence(seq):
			tensor = seq.header['_tensor']
			self.assertEqual(tensor['shape'],  [-1,1,2])
			self.assertEqual(tensor['dtype'],  'u8')
			self.assertEqual(tensor['labels'], ['time', 'pol', 'freq'])
			self.assertEqual(tensor['units'],  ['s', None, 'MHz'])
		def check_data(ispan, ospan):
			self.assertLessEqual(ispan.nframe, gulp_nframe)
			self.assertEqual(    ospan.nframe, ispan.nframe)
			self.assertEqual(ispan.data.shape, (ispan.nframe,1,2))
			self.assertEqual(ospan.data.shape, (ospan.nframe,1,2))
		with bfp.Pipeline() as pipeline:
			data = read_sigproc([self.fil_file], gulp_nframe)
			data = CallbackBlock(data, check_sequence, check_data)
			pipeline.run()
	def test_simple_copy(self):
		gulp_nframe = 101
		with bfp.Pipeline() as pipeline:
			data = read_sigproc([self.fil_file], gulp_nframe)
			data = copy(data)
			pipeline.run()
	def test_cuda_copy(self):
		gulp_nframe = 101
		with bfp.Pipeline() as pipeline:
			data = read_sigproc([self.fil_file], gulp_nframe)
			for _ in xrange(100):
				data = copy(data, space='cuda')
				data = copy(data, space='cuda_host')
			pipeline.run()
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
		with bfp.Pipeline() as pipeline:
			data = read_sigproc([self.fil_file], gulp_nframe)
			data = copy(data, space='cuda')
			data = transpose(data, ['pol', 'freq', 'time'])
			data = fdmt(data, max_dm=30.)
			data = CallbackBlock(data, check_sequence, check_data)
			data = transpose(data, ['time', 'pol', 'dispersion'])
			data = copy(data, space='cuda_host')
			pipeline.run()
	
