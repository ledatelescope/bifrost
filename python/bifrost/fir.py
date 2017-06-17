# -*- coding: utf-8 -*-

# Copyright (c) 2017, The Bifrost Authors. All rights reserved.
# Copyright (c) 2017, The University of New Mexico. All rights reserved.
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

from libbifrost import _bf, _check, _get, _fast_call, _string2space
from ndarray import asarray, zeros

import ctypes
import numpy as np

class FIR(object):
	def __init__(self):
		self.obj = _get(_bf.FIRCreate(), retarg=0)
	def __del__(self):
		if hasattr(self, 'obj') and bool(self.obj):
			_bf.FIRDestroy(self.obj)
		try:
			del self._prots
		except AttributeError:
			pass
	def init(self, ncoeff, decim, ntime, nstand, space='cuda'):
		space = _string2space(space)
		psize = None
		_check( _bf.FIRInit(self.obj, ncoeff, decim, ntime, nstand, space, 0, psize) )
	def set_coeffs(self, coeffs):
		_check( _bf.FIRSetCoeffs(self.obj, 
		                         asarray(coeffs).as_BFarray()) )
	def reset_state(self):
		_check( _bf.FIRResetState(self.obj) )
	def execute(self, idata, odata):
		# TODO: Work out how to integrate CUDA stream
		_check( _bf.FIRExecute(self.obj,
		                        asarray(idata).as_BFarray(),
		                        asarray(odata).as_BFarray()) )
		return odata