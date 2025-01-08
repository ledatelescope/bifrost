# Copyright (c) 2018-2025, The Bifrost Authors. All rights reserved.
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

from bifrost.libbifrost import _bf, _check, _get, BifrostObject, _string2space
from bifrost.ndarray import ndarray, zeros, empty_like, asarray, memset_array, copy_array

from bifrost.fft import Fft
from bifrost.map import map
from bifrost.reduce import reduce

import ctypes
import numpy

import time

from bifrost.device import stream_synchronize as BFSync

class Orville(BifrostObject):
    def __init__(self):
        BifrostObject.__init__(self, _bf.bfOrvilleCreate, _bf.bfOrvilleDestroy)
    def init(self, positions, weights, kernel, gridsize, gridres, gridwres=0.5, oversample=1, polmajor=True, space='cuda'):
        bspace = _string2space(space)
        psize = None
        _check( _bf.bfOrvilleInit(self.obj, 
                                  asarray(positions).as_BFarray(), 
                                  asarray(weights).as_BFarray(), 
                                  asarray(kernel).as_BFarray(),
                                  gridsize,
                                  gridres,
                                  gridwres,
                                  oversample,
                                  polmajor,
                                  bspace,
                                  0,
                                  psize) )
        
        # Cache the kernel for later
        self._kernel = kernel
        
        # Cache some metdata for later
        self._origsize = gridsize
        self._res = gridres
        self._maxsupport = kernel.size/oversample
        self._oversample = oversample
        self._dtype = kernel.dtype
        self._space = space
        self._update_beam = True
        
        # Build the image correction kernel and the gridding kernels
        self._set_image_correction_kernel()
        self._set_projection_kernels()
        
    def __del__(self):
        try:
            del self._subgrid
            del self._stcgrid
            del self._w_kernels
            del self._fft_im
            del self._fft_wp
        except AttributeError:
            pass
        
        BifrostObject.__del__(self)
        
    def _get_gridding_setup(self, force=False):
        try:
            self._ntimechan
            if force:
                raise AttributeError
        except AttributeError:
            # Get the  projection setup information
            ntimechan, gridsize, npol, nplane = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
            midpoints = zeros(128, dtype='f32', space='system')
            _check( _bf.bfOrvilleGetProjectionSetup(self.obj,
                                                    ctypes.byref(ntimechan),
                                                    ctypes.byref(gridsize),
                                                    ctypes.byref(npol),
                                                    ctypes.byref(nplane),
                                                    asarray(midpoints).as_BFarray()) )
            ntimechan, gridsize, npol, nplane = ntimechan.value, gridsize.value, npol.value, nplane.value
            midpoints = midpoints[:nplane]
            self._ntimechan = ntimechan
            self._gridsize = gridsize
            self._npol = npol
            self._nplane = nplane
            self._midpoints = midpoints
            print("Working with %i planes" % self._nplane)
            
    def _set_image_correction_kernel(self, force=False):
        self._get_gridding_setup(force=force)
        
        # Cleanup the old image correction kernel
        try:
            del self._img_correction
        except AttributeError:
            pass
            
        # Get the new image correction kernel
        corr = numpy.zeros(self._gridsize*self._oversample, dtype=self._dtype)
        c = corr.size//2
        d = self._kernel.size//2
        corr[c-d:c+d] = self._kernel.copy(space='system')
        corr[:c-d] = numpy.arange(c-d) * corr[c-d]/(c-d)
        corr[c+d:] = corr[c-d] - numpy.arange(c-d) * corr[c-d]/(c-d)
        corr = numpy.fft.ifftshift(corr)
        corr = numpy.fft.ifft(corr).real
        corr = numpy.fft.fftshift(corr)
        corr = corr[corr.size//2-self._gridsize//2-self._gridsize%2:corr.size//2+self._gridsize//2] * self._oversample
        corr = numpy.fft.fftshift(corr)
        self._img_correction = ndarray(corr, dtype='f32', space='cuda')
        
    def _get_lm(self, gridsize, gridres):
        m,l = numpy.indices((gridsize,gridsize))
        l,m = numpy.where(l > gridsize//2, gridsize-l, -l), numpy.where(m > gridsize//2, m-gridsize, m)
        l,m = l.astype(numpy.float32)/gridsize/gridres, m.astype(numpy.float32)/gridsize/gridres
        return l,m
        
    def _set_projection_kernels(self):
        self._get_gridding_setup()
        
        # Set the sub and stacked grids
        try:
            del self._subgrid
            del self._stcgrid
        except AttributeError:
            pass
        self._subgrid = zeros((self._nplane,self._ntimechan,self._npol,self._gridsize,self._gridsize), 
                              dtype=self._dtype, 
                              space=self._space)
        self._stcgrid = zeros((1,self._ntimechan,self._npol,self._gridsize,self._gridsize), 
                              dtype=self._dtype, 
                              space=self._space)
        
        # Set the w projection kernels
        try:
            del self._w_kernels
        except AttributeError:
            pass
        w_kernels = numpy.zeros((self._nplane,self._gridsize,self._gridsize), dtype=self._dtype)
        l,m = self._get_lm(self._gridsize, self._res)
        theta = numpy.sqrt(1.0 + 0.0j - l**2 - m**2)
        for i,avg_w in enumerate(self._midpoints):
            sub_w_kernel = numpy.exp(2j*numpy.pi*avg_w*(1.0-theta)) / self._gridsize**2 / self._gridsize**2
            w_kernels[i,:,:] = sub_w_kernel
        self._w_kernels = ndarray(w_kernels, space=self._space)
        
    def set_positions(self, positions):
        _check( _bf.bfOrvilleSetPositions(self.obj, 
                                          asarray(positions).as_BFarray()) )
        self._set_projection_kernels(force=True)
        self._update_beam = True
        
    def set_weights(self, weights):
        weights = numpy.fft.fftshift(weights)
        _check( _bf.bfOrvilleSetWeights(self.obj, 
                                       asarray(weights).as_BFarray()) )
        self._update_beam = True
        
    def set_kernel(self, kernel):
        self._kernel = kernel
        _check( _bf.bfOrvilleSetKernel(self.obj, 
                                       asarray(kernel).as_BFarray()) )
        self._set_image_correction_kernel()
        self._update_beam = True
        
    def execute(self, idata, odata, weighting='natural'):
        # TODO: Work out how to integrate CUDA stream
        
        if self._update_beam:
            bidata = empty_like(idata)
            map("bdata = Complex<float>(1,0)", {'bdata': bidata}, shape=bidata.shape)
            self._raw_beam = zeros(shape=(1,self._ntimechan,self._npol,self._gridsize,self._gridsize),
                                   dtype=self._stcgrid.dtype, space='cuda')
            
            self._update_beam = False
            try:
                self._raw_beam = self.execute(bidata, self._raw_beam, weighting='_raw_beam')
            except Exception as e:
                self._update_beam = True
                raise e
                
            rb = self._raw_beam.reshape(1,self._ntimechan,self._npol,self._gridsize*self._gridsize)
            self._wavg = zeros(shape=(1,self._ntimechan,self._npol,1),
                               dtype='cf32', space='cuda')
            reduce(rb, self._wavg, op='sum')
            self._w2avg = zeros(shape=(1,self._ntimechan,self._npol,1),
                               dtype='f32', space='cuda')
            reduce(rb, self._w2avg, op='pwrsum')
            
        # Zero out the subgrids
        memset_array(self._subgrid, 0)
        self._stcgrid = self._stcgrid.reshape(1,self._ntimechan,self._npol,self._gridsize,self._gridsize)
        
        # Grid
        _check( _bf.bfOrvilleExecute(self.obj,
                                     asarray(idata).as_BFarray(),
                                     asarray(self._subgrid).as_BFarray()) )
        
        if self._subgrid.shape[0] > 1:
            # W project
            ## FFT
            self._subgrid = self._subgrid.reshape(-1,self._gridsize,self._gridsize)
            try:
                self._fft_wp.execute(self._subgrid, self._subgrid, inverse=True)
            except AttributeError:
                self._fft_wp = Fft()
                self._fft_wp.init(self._subgrid, self._subgrid, axes=(1,2))
                self._fft_wp.execute(self._subgrid, self._subgrid, inverse=True)
            self._subgrid = self._subgrid.reshape(self._nplane,self._ntimechan,self._npol,self._gridsize,self._gridsize)
                
            ## Project
            map('a(w,c,p,i,j) *= b(w,i,j)', 
                {'a':self._subgrid, 'b':self._w_kernels}, 
                axis_names=('w','c','p','i','j'), shape=self._subgrid.shape)
            
            ## IFFT
            self._subgrid = self._subgrid.reshape(-1,self._gridsize,self._gridsize)
            self._fft_wp.execute(self._subgrid, self._subgrid, inverse=False)
            self._subgrid = self._subgrid.reshape(self._nplane,self._ntimechan,self._npol,self._gridsize,self._gridsize)
            
            # Stack
            reduce(self._subgrid, self._stcgrid, op='sum')
            
        else:
            copy_array(self._stcgrid, self._subgrid)
            
        # Apply the weighting
        if weighting == '_raw_beam':
            copy_array(odata, self._stcgrid)
            return odata
            
        elif weighting == 'natural':
            pass
            
        elif weighting == 'uniform':
           map("""
                auto weight = bdata(0,i,j,k,l).real;
                if( weight != 0.0 ) {
                  idata(0,i,j,k,l) = idata(0,i,j,k,l) / weight;
                }
                """,
                {'idata': self._stcgrid, 'bdata': self._raw_beam},
                axis_names=('i','j','k','l'), shape=self._stcgrid.shape[1:]
               )
                
        elif weighting.startswith('briggs'):
            R = float(weighting.split('@', 1)[1])
            map("""
                auto weight = bdata(0,i,j,k,l).real;
                auto S2 = 25.0 * powf(10.0, -2*{R}) / (w2avg(0,i,j,0) / wavg(0,i,j,0));
                if( weight != 0.0 ) {{
                    idata(0,i,j,k,l) = idata(0,i,j,k,l) / (1.0 + weight * S2);
                }}
                """.format(R=R),
                {'idata': self._stcgrid, 'bdata': self._raw_beam, 'wavg': self._wavg, 'w2avg': self._w2avg},
                axis_names=('i','j','k','l'), shape=self._stcgrid.shape[1:]
               )
                
        else:
            raise ValueError("Unknown weighting '%s'" % weighting)
            
        # (u,v) plane -> image
        ## IFFT
        self._stcgrid = self._stcgrid.reshape(-1,self._gridsize,self._gridsize)
        try:
            self._fft_im.execute(self._stcgrid, self._stcgrid, inverse=True)
        except AttributeError:
            self._fft_im = Fft()
            self._fft_im.init(self._stcgrid, self._stcgrid, axes=(1,2))
            self._fft_im.execute(self._stcgrid, self._stcgrid, inverse=True)
            
        # Correct for the kernel, shift, and accumulate onto odata
        oshape = odata.shape
        odata = odata.reshape(-1,odata.shape[-2],odata.shape[-1])
        padding = self._gridsize - self._origsize
        offset = padding//2# - padding%2
        map("""
            auto k = i + {offset} - {gridsize}/2;
            auto l = j + {offset} - {gridsize}/2;
            if( k < 0 ) k += {gridsize};
            if( l < 0 ) l += {gridsize};
            odata(c,i,j) += idata(c,k,l).real / (corr(k) * corr(l));
            """.format(gridsize=self._gridsize, offset=offset), 
            {'odata':odata, 'idata':self._stcgrid, 'corr':self._img_correction}, 
            axis_names=('c','i','j'), shape=odata.shape)
        odata = odata.reshape(oshape)
        
        return odata
