
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

from bifrost.pipeline import TransformBlock
from bifrost.linalg import LinAlg
from bifrost.DataType import DataType
import bifrost as bf

from copy import deepcopy
from math import sqrt

class ConvertVisibilitiesBlock(TransformBlock):
    def __init__(self, iring, fmt,
                 *args, **kwargs):
        super(ConvertVisibilitiesBlock, self).__init__(iring, *args, **kwargs)
        self.ofmt = fmt
    def define_valid_input_spaces(self):
        return ('cuda',)
    def on_sequence(self, iseq):
        ihdr = iseq.header
        itensor = ihdr['_tensor']
        ilabels = itensor['labels']
        assert(ilabels[0] == 'time')
        ohdr = deepcopy(ihdr)
        otensor = ohdr['_tensor']

        if ilabels[1:] == ['freq', 'station_i', 'pol_i', 'station_j', 'pol_j']:
            nchan, nstand, npol, nstand_j, npol_j = itensor['shape'][1:]
            assert(nstand_j == nstand)
            assert(  npol_j == npol)
            self.ifmt = 'matrix'
            if self.ofmt == 'matrix':
                ohdr['matrix_fill_mode'] = 'hermitian'
            elif self.ofmt == 'storage':
                nbaseline = nstand*(nstand+1)//2
                del ohdr['matrix_fill_mode']
                otensor['labels'] = ['time', 'baseline', 'freq', 'stokes']
                otensor['shape']  = [-1, nbaseline, nchan, npol*npol]
                time_units, freq_units, stand_units, pol_units, _, _ = itensor['units']
                otensor['units']  = [time_units, None, freq_units, ('I', 'Q', 'U', 'V')]
            else:
                raise NotImplementedError("Unsupported conversion from " +
                                          self.ifmt + " to " + self.ofmt)
        elif ilabels[1:] == ['baseline', 'freq', 'stokes']:
            nbaseline, nchan, nstokes = itensor['shape'][1:]
            assert(nstokes == 1 or nstokes == 4)
            npol = 1 if nstokes == 1 else 2
            nstand = int(sqrt(8 * nbaseline + 1) - 1) // 2
            time_units, baseline_units, freq_units, stokes_units, = itensor['units']
            pol_units = ('X', 'Y') # TODO: Support L/R (using additional metadata?)
            self.ifmt = 'storage'
            if self.ofmt == 'matrix':
                otensor['labels'] = ['time', 'freq', 'station_i', 'pol_i', 'station_j', 'pol_j']
                otensor['shape']  = [-1, nchan, nstand, npol, nstand, npol]
                otensor['units']  = [time_units, freq_units, None, pol_units, None, pol_units]
        else:
            raise NotImplementedError("Cannot convert input from %s to %s"
                                      % (ilabels, self.ofmt))
        return ohdr
    def on_data(self, ispan, ospan):
        idata = ispan.data
        odata = ospan.data
        itype = DataType(idata.dtype)
        otype = DataType(odata.dtype)
        if self.ifmt == 'matrix' and self.ofmt == 'matrix':
            # Make a full-matrix copy of the lower-only input matrix
            # odata[t,c,i,p,j,q] = idata[t,c,i,p,j,q] (lower filled only)
            shape_nopols = list(idata.shape)
            del shape_nopols[5]
            del shape_nopols[3]
            idata = idata.view(itype.as_vector(2))
            odata = odata.view(otype.as_vector(2))
            bf.map(
                '''
                bool in_lower_triangle = (i > j);
                if( in_lower_triangle ) {
                    odata(t,c,i,0,j,0) = idata(t,c,i,0,j,0);
                    odata(t,c,i,1,j,0) = idata(t,c,i,1,j,0);
                } else {
                    auto x = idata(t,c,j,0,i,0);
                    auto y = idata(t,c,j,1,i,0);
                    auto x1 = x[1];
                    x[0] = x[0].conj();
                    x[1] = y[0].conj();
                    if( i != j ) {
                        y[0] = x1.conj();
                    }
                    y[1] = y[1].conj();
                    odata(t,c,i,0,j,0) = x;
                    odata(t,c,i,1,j,0) = y;
                }
                ''',
                   shape=shape_nopols, axis_names=['t', 'c', 'i', 'j'],
                   data={'idata': idata, 'odata': odata})
        elif self.ifmt == 'matrix' and self.ofmt == 'storage':
            assert(idata.shape[2] <= 2048)
            idata = idata.view(itype.as_vector(2))
            odata = odata.view(otype.as_vector(4))
            # TODO: Support L/R as well as X/Y pols
            bf.map('''
            // TODO: This only works up to 2048 in single-precision
            #define project_triangular(i, j) ((i)*((i)+1)/2 + (j))
            int i = int((sqrt(8.f*(b)+1)-1)/2);
            int j = b - project_triangular(i, 0);
            auto x = idata(t,c,i,0,j,0);
            auto y = idata(t,c,i,1,j,0);
            if( i == j ) {
                x[1] = y[0].conj();
            }
            idata_type::value_type eye(0, 1);
            auto I = (x[0] + y[1]);
            auto Q = (x[0] - y[1]);
            auto U = (x[1] + y[0]);
            auto V = (x[1] - y[0]) * eye;
            odata(t,b,c,0) = odata_type(I,Q,U,V);
            ''',
                   shape=odata.shape[:-1], axis_names=['t', 'b', 'c'],
                   data={'idata': idata, 'odata': odata},
                   block_shape=[64,8]) # TODO: Tune this
        #elif self.ifmt == 'matrix' and self.ofmt == 'triangular':
        elif self.ifmt == 'storage' and self.ofmt == 'matrix':
            oshape_nopols = list(odata.shape)
            del oshape_nopols[5]
            del oshape_nopols[3]
            idata = idata.view(itype.as_vector(4))
            odata = odata.view(otype.as_vector(2))
            bf.map('''
            bool in_upper_triangle = (i < j);
            auto b = in_upper_triangle ? j*(j+1)/2 + i : i*(i+1)/2 + j;
            auto IQUV = idata(t,b,c,0);
            auto I = IQUV[0], Q = IQUV[1], U = IQUV[2], V = IQUV[3];
            idata_type::value_type eye(0, 1);
            auto xx = 0.5f*(I + Q);
            auto xy = 0.5f*(U - V*eye);
            auto yx = 0.5f*(U + V*eye);
            auto yy = 0.5f*(I - Q);
            if( i == j ) {
                xy = yx.conj();
            }
            if( in_upper_triangle ) {
                auto tmp_xy = xy;
                xx = xx.conj();
                xy = yx.conj();
                yx = tmp_xy.conj();
                yy = yy.conj();
            }
            odata(t,c,i,0,j,0) = odata_type(xx, xy);
            odata(t,c,i,1,j,0) = odata_type(yx, yy);
            ''',
                   shape=oshape_nopols, axis_names=['t', 'c', 'i', 'j'],
                   data={'idata': idata, 'odata': odata},
                   block_shape=[64,8]) # TODO: Tune this
        else:
            raise NotImplementedError

def convert_visibilities(iring, fmt, *args, **kwargs):
    """Convert visibility data to a new format.

    Supported values of 'fmt' are:
      matrix, storage

    Args:
        iring (Ring or Block): Input data source.
        fmt (str): The desired output format: matrix, storage.
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.

    **Tensor semantics**::

        Input:  ['time', 'freq', 'station_i', 'pol_i', 'station_j', 'pol_j'], dtype = any complex, space = CUDA
        fmt = 'matrix' (produces a fully-filled matrix from a lower-filled one)
        Output: ['time', 'freq', 'station_i', 'pol_i', 'station_j', 'pol_j'], dtype = any complex, space = CUDA
        fmt = 'storage' (suitable for common on-disk data formats such as UVFITS, FITS-IDI, MS etc.)
        Output: ['time', 'baseline', 'freq', 'stokes'], dtype = any complex, space = CUDA

        Input:  ['time', 'baseline', 'freq', 'stokes'], dtype = any complex, space = CUDA
        fmt = 'matrix' (fully-filled matrix suitable for linear algebra operations)
        Output: ['time', 'freq', 'station_i', 'pol_i', 'station_j', 'pol_j'], dtype = any complex, space = CUDA

    Returns:
        ConvertVisibilitiesBlock: A new block instance.
    """
    return ConvertVisibilitiesBlock(iring, fmt, *args, **kwargs)
