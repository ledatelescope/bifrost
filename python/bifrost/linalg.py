
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

from libbifrost import _bf, _check, _get, BifrostObject
import ctypes
from ndarray import asarray

class LinAlg(BifrostObject):
    def __init__(self):
        BifrostObject.__init__(self, _bf.bfLinAlgCreate, _bf.bfLinAlgDestroy)
    def matmul(self, alpha, a, b, beta, c):
        """Computes:
          c = alpha*a.b + beta*c
        or if b is None:
          c = alpha*a.a^H + beta*c
        or if a is None:
          c = alpha*b^H.b + beta*c
        where '.' is matrix product and '^H' is Hermitian transpose.
        Multi-dimensional semantics are the same as numpy.matmul:
          The last two dims represent the matrix, and all other dims are
          used as batch dims to be matched or broadcast between a and b.
        """
        if alpha is None:
            alpha = 1.
        if beta is None:
            beta = 0.
        beta  = float(beta)
        alpha = float(alpha)
        a_array = asarray(a).as_BFarray() if a is not None else None
        b_array = asarray(b).as_BFarray() if b is not None else None
        c_array = asarray(c).as_BFarray()
        _check(_bf.bfLinAlgMatMul(self.obj,
                                  alpha,
                                  a_array,
                                  b_array,
                                  beta,
                                  c_array))
        return c
