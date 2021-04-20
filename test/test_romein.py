
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

from __future__ import print_function

import unittest
import numpy
import bifrost
from bifrost.romein import Romein
from bifrost.quantize import quantize
from bifrost.unpack import unpack
from bifrost.DataType import ci4

class RomeinTest(unittest.TestCase):
    def setUp(self):
        self.romein=Romein()
        numpy.random.seed(1337)
        
    def naive_romein(self,
                     grid_shape,
                     illum,
                     data,
                     xlocs,
                     ylocs,
                     zlocs,
                     ntime,
                     npol,
                     nchan,
                     ndata,
                     dtype=numpy.complex64):
        
        # Unpack the ci4 data to ci8, if needed
        if data.dtype == ci4:
            data_unpacked = bifrost.ndarray(shape=data.shape, dtype='ci8')
            unpack(data, data_unpacked)
            data = data_unpacked
            
        #Excruciatingly slow, but it's just for testing purposes...
        #Could probably use a blas based function for simplicity.
        grid = numpy.zeros(shape=grid_shape,dtype=dtype)
        for t in numpy.arange(ntime):
            for c in numpy.arange(nchan):
                for p in numpy.arange(npol):
                    for d in numpy.arange(ndata):
                        datapoint = data[t,c,p,d]
                        if data.dtype != dtype:
                            try:
                                datapoint = dtype(datapoint[0]+1j*datapoint[1])
                            except IndexError:
                                datapoint = dtype(datapoint)
                                
                        #if(d==128):
                        #    print(datapoint)
                        x_s = xlocs[t,c,p,d]
                        y_s = ylocs[t,c,p,d]
                        for y in numpy.arange(y_s,y_s+illum.shape[4]):
                            for x in numpy.arange(x_s,x_s+illum.shape[5]):
                                illump = illum[t,c,p,d,y-y_s,x-x_s]
                                grid[t,c,p,y,x] += datapoint * illump

        return grid
        
    def _create_illum(self, illum_size, data_size, ntime, npol, nchan, dtype=numpy.complex64):
        illum_shape = (ntime,nchan,npol,data_size,illum_size,illum_size)
        illum = numpy.ones(shape=illum_shape,dtype=dtype)
        illum = numpy.copy(illum,order='C')
        illum = bifrost.ndarray(illum)
        
        return illum
        
    def _create_locs(self, data_size, ntime, nchan, npol, loc_min, loc_max):
        ishape = (ntime,nchan,npol,data_size)
        xlocs = numpy.random.uniform(loc_min, loc_max, size=ishape)
        ylocs = numpy.random.uniform(loc_min, loc_max, size=ishape)
        zlocs = numpy.random.uniform(loc_min, loc_max, size=ishape)
        xlocs = numpy.copy(xlocs.astype(numpy.int32),order='C')
        ylocs = numpy.copy(ylocs.astype(numpy.int32),order='C')
        zlocs = numpy.copy(zlocs.astype(numpy.int32),order='C')
        xlocs = bifrost.ndarray(xlocs)
        ylocs = bifrost.ndarray(ylocs)
        zlocs = bifrost.ndarray(zlocs)
        locs = numpy.stack((xlocs, ylocs, zlocs))
        #locs = numpy.transpose(locs,(1,2,3,4,0))
        locs = numpy.copy(locs.astype(numpy.int32),order='C')
        locs = bifrost.ndarray(locs)
        
        return locs
        
    def _create_data(self, data_size, ntime, nchan, npol, dtype=numpy.complex64):
        ishape = (ntime,nchan,npol,data_size)
        data = numpy.zeros(shape=ishape,dtype=numpy.complex64)
        data_i = numpy.zeros(shape=(ntime,nchan,npol,data_size,2), dtype=numpy.complex64)
        data_i[:,:,:,:,0] = numpy.random.normal(0,1.0,size=(ntime,nchan,npol,data_size))
        data_i[:,:,:,:,1] = numpy.random.normal(0,1.0,size=(ntime,nchan,npol,data_size))
        data = 7*numpy.copy(data,order='C')
        data = bifrost.ndarray(data_i[...,0] + 1j * data_i[...,1])
        if dtype not in (numpy.complex64, 'cf32'):
            data_quantized = bifrost.ndarray(shape=data.shape, dtype=dtype)
            quantize(data, data_quantized)
            data = data_quantized
            
        return data
        
    def run_test(self, grid_size, illum_size, data_size, ntime, npol, nchan, polmajor, dtype=numpy.complex64, otype=numpy.complex64):
        
        gridshape = (ntime,nchan,npol,grid_size,grid_size)
        ishape = (ntime,nchan,npol,data_size)
        illum_shape = (ntime,nchan,npol,data_size,illum_size,illum_size)

        # Create grid and illumination pattern
        grid = numpy.zeros(shape=gridshape,dtype=otype)
        grid = numpy.copy(grid,order='C')
        grid = bifrost.ndarray(grid)
        illum = self._create_illum(illum_size, data_size, ntime, npol, nchan, dtype=otype)
        
        # Create data
        data = self._create_data(data_size, ntime, nchan, npol, dtype=dtype)
        
        # Create the locations
        locs = self._create_locs(data_size, ntime, nchan, npol, illum_size, grid_size-illum_size)
        
        # Grid using a naive method
        gridnaive = self.naive_romein(gridshape,illum,data,locs[0,:],locs[1,:],locs[2,:],ntime,npol,nchan,data_size,otype)

        # Transpose for non pol-major kernels
        if not polmajor:
            data=data.transpose((0,1,3,2)).copy()
            locs=locs.transpose((0,1,2,4,3)).copy()
            illum=illum.transpose((0,1,3,2,4,5)).copy()            
        
        grid = grid.copy(space='cuda')
        data = data.copy(space='cuda')
        illum = illum.copy(space='cuda')
        locs = locs.copy(space='cuda')
        self.romein.init(locs,illum, grid_size,polmajor=polmajor)
        self.romein.execute(data,grid)
        grid = grid.copy(space="system")
        
        # Compare the two methods
        numpy.testing.assert_allclose(grid,
                                      gridnaive,
                                      (1e-10 if otype == numpy.complex128 else 1e-4),
                                      (1e-11 if otype == numpy.complex128 else 1e-5))
        
    def run_kernel_test(self, grid_size, illum_size, data_size, ntime, npol, nchan, polmajor, dtype=numpy.complex64):
        TEST_SCALE = 2.0
        
        gridshape = (ntime,nchan,npol,grid_size,grid_size)
        ishape = (ntime,nchan,npol,data_size)
        illum_shape = (ntime,nchan,npol,data_size,illum_size,illum_size)

        # Create grid and illumination pattern
        grid = numpy.zeros(shape=gridshape,dtype=numpy.complex64)
        grid = numpy.copy(grid,order='C')
        grid = bifrost.ndarray(grid)
        illum = self._create_illum(illum_size, data_size, ntime, npol, nchan)
        
        # Create data
        data = self._create_data(data_size, ntime, nchan, npol, dtype=dtype)
        
        # Create the locations
        locs = self._create_locs(data_size, ntime, nchan, npol, illum_size, grid_size-illum_size)
        
        # Grid using a naive method
        gridnaive = self.naive_romein(gridshape,illum*TEST_SCALE,data,locs[0,:],locs[1,:],locs[2,:],ntime,npol,nchan,data_size)
        
        # Transpose for non pol-major kernels
        if not polmajor:
            data=data.transpose((0,1,3,2)).copy()
            locs=locs.transpose((0,1,2,4,3)).copy()
            illum=illum.transpose((0,1,3,2,4,5)).copy()            
        
        grid = grid.copy(space='cuda')
        data = data.copy(space='cuda')
        illum = illum.copy(space='cuda')
        locs = locs.copy(space='cuda')
        self.romein.init(locs,illum, grid_size,polmajor=polmajor)
        
        # Scale
        illum = illum.copy(space='system')
        illum *= TEST_SCALE
        illum = illum.copy(space='cuda')
        
        self.romein.set_kernels(illum)
        self.romein.execute(data,grid)
        grid = grid.copy(space="system")
        
        # Compare the two methods
        numpy.testing.assert_allclose(grid, gridnaive, 1e-4, 1e-5)
        
    def run_positions_test(self, grid_size, illum_size, data_size, ntime, npol, nchan, polmajor, dtype=numpy.complex64):
        TEST_OFFSET = 1
        
        gridshape = (ntime,nchan,npol,grid_size,grid_size)
        ishape = (ntime,nchan,npol,data_size)
        illum_shape = (ntime,nchan,npol,data_size,illum_size,illum_size)

        # Create grid and illumination pattern
        grid = numpy.zeros(shape=gridshape,dtype=numpy.complex64)
        grid = numpy.copy(grid,order='C')
        grid = bifrost.ndarray(grid)
        illum = self._create_illum(illum_size, data_size, ntime, npol, nchan)
        
        # Create data
        data = self._create_data(data_size, ntime, nchan, npol, dtype=dtype)
        
        # Create the locations
        locs = self._create_locs(data_size, ntime, nchan, npol, illum_size, grid_size-illum_size)
        
        # Grid using a naive method
        gridnaive = self.naive_romein(gridshape,illum,data,locs[0,:]+TEST_OFFSET,locs[1,:]+TEST_OFFSET,locs[2,:]+TEST_OFFSET,ntime,npol,nchan,data_size)
        
        # Transpose for non pol-major kernels
        if not polmajor:
            data=data.transpose((0,1,3,2)).copy()
            locs=locs.transpose((0,1,2,4,3)).copy()
            illum=illum.transpose((0,1,3,2,4,5)).copy()            
        
        grid = grid.copy(space='cuda')
        data = data.copy(space='cuda')
        illum = illum.copy(space='cuda')
        locs = locs.copy(space='cuda')
        self.romein.init(locs,illum, grid_size,polmajor=polmajor)
        
        # Offset
        locs = locs.copy(space='system')
        locs += TEST_OFFSET
        locs = locs.copy(space='cuda')
        
        self.romein.set_positions(locs)
        self.romein.execute(data,grid)
        grid = grid.copy(space="system")
        
        # Compare the two methods
        numpy.testing.assert_allclose(grid, gridnaive, 1e-4, 1e-5)
        
    def test_ntime8_nchan2_npol3_gridsize64_illumsize3_datasize256_pm(self):
        self.run_test(grid_size=64, illum_size=3, data_size=256, ntime=8, npol=3, nchan=2,polmajor=True)
    def test_ntime8_nchan2_npol3_gridsize64_illumsize3_datasize256(self):
        self.run_test(grid_size=64, illum_size=3, data_size=256, ntime=8, npol=3, nchan=2, polmajor=False)
    def test_ntime2_nchan2_npol2_gridsize64_illumsize7_datasize256_pm(self):
        self.run_test(grid_size=64, illum_size=7, data_size=256, ntime=2, npol=2, nchan=2,polmajor=True)
    def test_ntime2_nchan2_npol2_gridsize64_illumsize7_datasize256(self):
        self.run_test(grid_size=64, illum_size=7, data_size=256, ntime=2, npol=2, nchan=2,polmajor=False)
    def test_ntime2_nchan2_npol2_gridsize128_illumsize3_datasize256_pm(self):
        self.run_test(grid_size=128, illum_size=3, data_size=256, ntime=2, npol=2, nchan=2,polmajor=True)
    def test_ntime2_nchan2_npol2_gridsize128_illumsize3_datasize256(self):
        self.run_test(grid_size=128, illum_size=3, data_size=256, ntime=2, npol=2, nchan=2,polmajor=False)
    def test_ntime_16_nchan4_npol2_gridsize64_illumsize5_datasize256_pm(self):
        self.run_test(grid_size=64, illum_size=5, data_size=256, ntime=16, npol=2, nchan=4,polmajor=True) 
    def test_ntime_32_nchan2_npol2_gridsize32_illumsize3_datasize256(self):
        self.run_test(grid_size=32, illum_size=3, data_size=256, ntime=32, npol=2, nchan=2,polmajor=False)
    def test_ntime_32_nchan2_npol2_gridsize32_illumsize3_datasize256_out_cf64(self):
        self.run_test(grid_size=32, illum_size=3, data_size=256, ntime=32, npol=2, nchan=2,polmajor=False,otype=numpy.complex128)
    def test_ntime_32_nchan2_npol2_gridsize32_illumsize3_datasize256_ci4(self):
        self.run_test(grid_size=32, illum_size=3, data_size=256, ntime=32, npol=2, nchan=2,polmajor=False, dtype='ci4')
    def test_ntime_32_nchan2_npol2_gridsize32_illumsize3_datasize256_ci8(self):
        self.run_test(grid_size=32, illum_size=3, data_size=256, ntime=32, npol=2, nchan=2,polmajor=False, dtype='ci8')
    def test_ntime_32_nchan2_npol2_gridsize32_illumsize3_datasize256_ci16(self):
        self.run_test(grid_size=32, illum_size=3, data_size=256, ntime=32, npol=2, nchan=2,polmajor=False, dtype='ci16')
    def test_ntime_32_nchan2_npol2_gridsize32_illumsize3_datasize256_ci32(self):
        self.run_test(grid_size=32, illum_size=3, data_size=256, ntime=32, npol=2, nchan=2,polmajor=False, dtype='ci32')
    def test_set_kernels_pm(self):
        self.run_kernel_test(grid_size=64, illum_size=3, data_size=256, ntime=8, npol=3, nchan=2,polmajor=True)
    def test_set_kernels(self):
        self.run_kernel_test(grid_size=64, illum_size=3, data_size=256, ntime=8, npol=3, nchan=2,polmajor=False)
    def test_set_positions_pm(self):
        self.run_positions_test(grid_size=64, illum_size=3, data_size=256, ntime=8, npol=3, nchan=2,polmajor=True)
    def test_set_positions(self):
        self.run_positions_test(grid_size=64, illum_size=3, data_size=256, ntime=8, npol=3, nchan=2,polmajor=False)


