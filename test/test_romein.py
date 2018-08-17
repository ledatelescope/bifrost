
# Copyright (c) 2018, The Bifrost Authors. All rights reserved.
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
import numpy
import bifrost
from bifrost.romein import romein_float
import matplotlib.pyplot as plt

class RomeinTest(unittest.TestCase):
    def naive_romein(self,
                     grid_shape,
                     illum,
                     data,
                     xlocs,
                     ylocs,
                     zlocs,
                     ntime,
                     npol,
                     nchan):
        #Excruciatingly slow, but it's just for testing purposes...
        #Could probably use a blas based function for simplicity.
        grid = numpy.zeros(shape=grid_shape,dtype=numpy.complex64)
        for t in numpy.arange(ntime):
            for c in numpy.arange(nchan):
                for p in numpy.arange(npol):
                    for d in numpy.arange(data.shape[-1]):

                        datapoint = data[t,c,p,d]
                        #if(d==128):
                        #    print(datapoint)
                        x_s = xlocs[t,c,p,d]
                        y_s = ylocs[t,c,p,d]
                        for y in numpy.arange(y_s,y_s+illum.shape[0]):
                            for x in numpy.arange(x_s,x_s+illum.shape[1]):
                                illump = illum[y-y_s,x-x_s]
                                grid[t,c,p,y,x] += datapoint * illump

        return grid

    
    def run_test(self, grid_size, illum_size, data_size, ntime, npol, nchan):
        
        gridshape = (ntime,nchan,npol,grid_size,grid_size)
        ishape = (ntime,nchan,npol,data_size)
        illum_shape = (illum_size,illum_size)

        # Create grid and illumination pattern
        grid = numpy.zeros(shape=gridshape,dtype=numpy.complex64)
        grid = numpy.copy(grid,order='C')
        grid = bifrost.ndarray(grid)
        
        illum = numpy.ones(shape=illum_shape,dtype=numpy.complex64)
        illum = numpy.copy(illum,order='C')
        illum = bifrost.ndarray(illum)

        # Create data

        data = numpy.zeros(shape=ishape,dtype=numpy.complex64)
        data_i = numpy.zeros(shape=(ntime,nchan,npol,data_size,2), dtype=numpy.complex64)
        data_i[:,:,:,:,0] = numpy.random.normal(0,1.0,size=(ntime,nchan,npol,data_size))
        data_i[:,:,:,:,1] = numpy.random.normal(0,1.0,size=(ntime,nchan,npol,data_size))
        data = numpy.copy(data,order='C')
        data = bifrost.ndarray(data_i[...,0] + 1j * data_i[...,1])


        xlocs = numpy.random.uniform(illum_size, grid_size-illum_size,size=ishape)
        ylocs = numpy.random.uniform(illum_size, grid_size-illum_size,size=ishape)
        zlocs = numpy.random.uniform(illum_size, grid_size-illum_size,size=ishape)
        xlocs = numpy.copy(xlocs.astype(numpy.int32),order='C')
        ylocs = numpy.copy(ylocs.astype(numpy.int32),order='C')
        zlocs = numpy.copy(zlocs.astype(numpy.int32),order='C')
        xlocs = bifrost.ndarray(xlocs)
        ylocs = bifrost.ndarray(ylocs)
        zlocs = bifrost.ndarray(zlocs)

        gridnaive = self.naive_romein(gridshape,illum,data,xlocs,ylocs,zlocs,ntime,npol,nchan)
        
        grid = grid.copy(space='cuda')
        data = data.copy(space='cuda')
        illum = illum.copy(space='cuda')
        xlocs = xlocs.copy(space='cuda')
        ylocs = ylocs.copy(space='cuda')
        zlocs = zlocs.copy(space='cuda')
        
        grid = romein_float(data,grid,illum,xlocs,ylocs,zlocs,illum_size,grid_size,data_size,ntime*npol*nchan)
        grid = grid.copy(space="system")
#        diff = grid - gridnaive
        
        diffs = numpy.sum(abs(grid.flatten()-gridnaive.flatten()))
        if diff > 0.01:
            raise ValueError("Large difference between naive romein and CUDA implementation.")
        
    def test_ntime8_nchan2_npol2_gridsize128_illumsize3_datasize256(self):
        self.run_test(grid_size=128, illum_size=3, data_size=256, ntime=8, npol=2, nchan=2)

    def test_ntime32_nchan64_npol2_gridsize128_illumsize3_datasize256(self):
        self.run_test(grid_size=128, illum_size=3, data_size=256, ntime=32, npol=2, nchan=64)
        
    def test_ntime512_nchan4_npol2_gridsize128_illumsize3_datasize256(self):
        self.run_test(grid_size=128, illum_size=3, data_size=256, ntime=512, npol=2, nchan=4)
        
    def test_ntime1024_nchan2_npol2_gridsize128_illumsize3_datasize256(self):
        self.run_test(grid_size=128, illum_size=3, data_size=256, ntime=1024, npol=2, nchan=2)

    def test_ntime_1024_nchan2_npol2_gridsize256_illumsize10_datasize256(self):
        self.run_test(grid_size=256, illum_size=10, data_size=256, ntime=1024, npol=2, nchan=2)



        