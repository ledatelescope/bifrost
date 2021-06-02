
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
from __future__ import absolute_import

from bifrost.libbifrost import _bf, _check
from bifrost.ndarray import asarray

REDUCE_MAP = {
    'sum':       _bf.BF_REDUCE_SUM,
    'mean':      _bf.BF_REDUCE_MEAN,
    'min':       _bf.BF_REDUCE_MIN,
    'max':       _bf.BF_REDUCE_MAX,
    'stderr':    _bf.BF_REDUCE_STDERR,
    'pwrsum':    _bf.BF_REDUCE_POWER_SUM,
    'pwrmean':   _bf.BF_REDUCE_POWER_MEAN,
    'pwrmin':    _bf.BF_REDUCE_POWER_MIN,
    'pwrmax':    _bf.BF_REDUCE_POWER_MAX,
    'pwrstderr': _bf.BF_REDUCE_POWER_STDERR,
}

def reduce(idata, odata, op='sum'):
    if op not in REDUCE_MAP:
        raise ValueError("Invalid reduce op: " + str(op))
    op = REDUCE_MAP[op]
    _check(_bf.bfReduce(asarray(idata).as_BFarray(),
                        asarray(odata).as_BFarray(),
                        op))
