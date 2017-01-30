
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
from bifrost.fdmt import Fdmt

class FdmtTest(unittest.TestCase):
	def test_fdmt(self):
		fdmt = Fdmt()
		ntime     = 1024
		nchan     = 128
		max_delay = 200
		f0        = 1000.
		bw        = 400.
		df        = bw / nchan
		exponent  = -2.0
		fdmt.init(nchan, max_delay, f0, df, exponent, 'cuda')
		idata = bf.asarray(np.random.normal(size=(nchan,ntime)).astype(np.float32), space='cuda')
		
		odata1 = bf.asarray(-999*np.ones((max_delay,ntime), np.float32), space='cuda')
		fdmt.execute(idata, odata1)
		odata1 = odata1.copy('system')
		self.assertEqual(odata1.min(), -999)
		# TODO: Need better tests
		self.assertLess(odata1.max(), 100.)
		
		odata2 = bf.asarray(-999*np.ones((max_delay,ntime), np.float32), space='cuda')
		workspace_size = fdmt.get_workspace_size(idata, odata2)
		self.assertEqual(workspace_size, 3293184)
		workspace = bf.asarray(np.empty(workspace_size, np.uint8), space='cuda')
		workspace_ptr = workspace.ctypes.data
		fdmt.execute_workspace(idata, odata2, workspace_ptr, workspace_size)
		odata2 = odata2.copy('system')
		np.testing.assert_equal(odata1, odata2)
