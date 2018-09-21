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

from libbifrost import _bf, _check, _get, BifrostObject
import ctypes
from ndarray import asarray

class Romein(BifrostObject):
    def __init__(self):
        BifrostObject.__init__(self, _bf.bfRomeinCreate, _bf.bfRomeinDestroy)
    def init(self, positions, kernels, ngrid):
        _check( _bf.bfRomeinInit(self.obj, 
                                 asarray(positions).as_BFarray(), 
                                 asarray(kernels).as_BFarray(), 
                                 ngrid) )
    def set_positions(self, positions):
        _check( _bf.bfRomeinSetPositions(self.obj, 
                                         asarray(positions).as_BFarray()) )
    def set_kernels(self, kernels):
        _check( _bf.bfRomeinSetKernels(self.obj, 
                                       asarray(kernels).as_BFarray()) )
    def execute(self, idata, odata):
        # TODO: Work out how to integrate CUDA stream
        _check( _bf.bfRomeinExecute(self.obj,
                                    asarray(idata).as_BFarray(),
                                    asarray(odata).as_BFarray()) )
        return odata


def romein_float(data,
                 grid,
                 kernel,
                 xlocs,
                 ylocs,
                 zlocs,
                 kern_size,
                 grid_size,
                 data_size,
                 nbatch):
    """ 
        Convolves data onto grid using kernel.
    
        Kernel size is same for all data. I want to
        extend it to different kernels for w-projection
        effects etc.
    """
    grid_array = asarray(grid).as_BFarray() 
    data_array = asarray(data).as_BFarray() 
    illum_array = asarray(kernel).as_BFarray() 
    xlocs_array = asarray(xlocs).as_BFarray() 
    ylocs_array = asarray(ylocs).as_BFarray() 
    zlocs_array = asarray(zlocs).as_BFarray() 
    _check(_bf.romein_float(data_array,
                            grid_array,
                            illum_array,
                            xlocs_array,
                            ylocs_array,
                            zlocs_array,
                            kern_size,
                            grid_size,
                            data_size,
                            nbatch))
    return grid
