
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

from __future__ import absolute_import

import bifrost as bf
from bifrost.pipeline import TransformBlock
from bifrost.DataType import DataType

from copy import deepcopy

class DetectBlock(TransformBlock):
    def __init__(self, iring, mode, axis=None,
                 *args, **kwargs):
        super(DetectBlock, self).__init__(iring, *args, **kwargs)
        self.specified_axis = axis
        self.mode = mode.lower()
    def define_valid_input_spaces(self):
        """Return set of valid spaces (or 'any') for each input"""
        return ('cuda',)
    def on_sequence(self, iseq):
        ihdr = iseq.header
        itensor = ihdr['_tensor']
        itype = DataType(itensor['dtype'])
        if not itype.is_complex:
            raise TypeError("Input data must be complex")
        self.axis = self.specified_axis
        if 'labels' not in itensor.keys() and self.axis is None:
            raise TypeError("Polarization (pol) index must be labelled, or axis must be set manually")
        elif (self.axis is None and
              self.mode != 'scalar' and
              'pol' in itensor['labels']):
            self.axis = itensor['labels'].index('pol')
        elif isinstance(self.axis, basestring):
            self.axis = itensor['labels'].index(self.axis)
        # Note: axis may be None here, which indicates single-pol mode
        ohdr = deepcopy(ihdr)
        otensor = ohdr['_tensor']
        if self.axis is not None:
            self.npol = otensor['shape'][self.axis]
            if self.npol not in [1, 2]:
                raise ValueError("Axis must have length 1 or 2")
            if self.mode == 'stokes' and self.npol == 2:
                otensor['shape'][self.axis] = 4
            if 'labels' in otensor:
                otensor['labels'][self.axis] = 'pol'
        else:
            self.npol = 1
        if self.mode == 'jones' and self.npol == 2:
            otype = itype
        else:
            otype = itype.as_real()
        otensor['dtype'] = otype.as_floating_point()
        return ohdr
    def on_data(self, ispan, ospan):
        idata = ispan.data
        odata = ospan.data
        if self.npol == 1:
            bf.map("b = Complex<b_type>(a).mag2()", {'a': idata, 'b': odata})
        else:
            shape = idata.shape[:self.axis] + idata.shape[self.axis + 1:]
            inds = ['i%i' % i for i in xrange(idata.ndim)]
            inds[self.axis] = '%i'
            inds_pol = ','.join(inds)
            inds_ = [inds_pol % i for i in xrange(4)]
            inds = inds[:self.axis] + inds[self.axis + 1:]
            if self.mode == 'jones':
                func = """
                b_type x = a(%s);
                b_type y = a(%s);
                b(%s).assign(x.mag2(), y.mag2());
                b(%s) = x*y.conj();
                """ % (inds_[0], inds_[1], inds_[0], inds_[1])
            elif self.mode == 'stokes':
                func = """
                Complex<b_type> x = a(%s);
                Complex<b_type> y = a(%s);
                auto xx = x.mag2();
                auto yy = y.mag2();
                auto xy = x*y.conj();
                b(%s) = xx + yy;
                b(%s) = xx - yy;
                b(%s) =  2*xy.real;
                b(%s) = -2*xy.imag;
                """ % (inds_[0], inds_[1],
                       inds_[0], inds_[1], inds_[2], inds_[3])
            bf.map(func, shape=shape, axis_names=inds,
                   data={'a': ispan.data, 'b': ospan.data})

def detect(iring, mode, axis=None, *args, **kwargs):
    """Apply square-law detection to create polarization products.

    Args:
        iring (Ring or Block): Input data source.
        mode (string):

           ``'scalar': x   -> real x.x*``

           ``'jones':  x,y -> complex x.x* + 1j*y.y*, x.y*``

           ``'stokes': x,y -> real I, Q, U, V``

        axis: Integer or string specifying the polarization axis. Defaults to
                'pol'. Not used if mode = 'scalar'.
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.

    **Tensor semantics**::

        Input:  [..., 'pol', ...], dtype = any complex, space = CUDA
        Output: [..., 'pol', ...], dtype = real or complex, space = CUDA

    Returns:
        DetectBlock: A new block instance.
    """
    return DetectBlock(iring, mode, axis, *args, **kwargs)
