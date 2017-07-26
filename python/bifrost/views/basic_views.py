
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

from bifrost.pipeline import block_view
from bifrost.DataType import DataType
from bifrost.units import convert_units
from numpy import isclose
from copy import deepcopy


def custom(block, hdr_transform):
    """An alias to `bifrost.pipeline.block_view`
    """
    return block_view(block, hdr_transform)

def rename_axis(block, old, new):
    def header_transform(hdr, old=old, new=new):
        axis = hdr['_tensor']['labels'].index(old)
        hdr['_tensor']['labels'][axis] = new
        return hdr
    return block_view(block, header_transform)

def reinterpret_axis(block, axis, label, scale=None, units=None):
    """ Manually reinterpret the scale and/or units on an axis """
    def header_transform(hdr, axis=axis, label=label, scale=scale, units=units):
        tensor = hdr['_tensor']
        if isinstance(axis, basestring):
            axis = tensor['labels'].index(axis)
        if label is not None:
            tensor['labels'][axis] = label
        if scale is not None:
            tensor['scales'][axis] = scale
        if units is not None:
            tensor['units'][axis] = units
        return hdr
    return block_view(block, header_transform)

def reverse_scale(block, axis):
    """ Manually reverse the scale factor on a given axis"""
    def header_transform(hdr, axis=axis):
        tensor = hdr['_tensor']
        if isinstance(axis, basestring):
            axis = tensor['labels'].index(axis)
            tensor['scales'][axis][1] *= -1
        return hdr
    return block_view(block, header_transform)

def expand_dims(block, axis, label, scale=None, units=None):
    """Add an extra dimension to the frame

    As in, if the shape is [-1, 3, 2], then
    selecting axis=1 would change the shape to be
    [-1, 3, 1, 2].
    """
    def header_transform(hdr, axis=axis, label=label, scale=scale, units=units):
        tensor = hdr['_tensor']
        if isinstance(axis, basestring):
            axis = tensor['labels'].index(axis)
        tensor['shape'].insert(axis, 1)
        tensor['labels'].insert(axis, label)
        tensor['scales'].insert(axis, scale)
        tensor['units'].insert(axis, units)
        return hdr
    return block_view(block, header_transform)

def astype(block, dtype):
    def header_transform(hdr, new_dtype=dtype):
        tensor = hdr['_tensor']
        old_dtype = tensor['dtype']
        old_itemsize = DataType(old_dtype).itemsize
        new_itemsize = DataType(new_dtype).itemsize
        old_axissize = old_itemsize * tensor['shape'][-1]
        if old_axissize % new_itemsize:
            raise ValueError("New type not compatible with data shape")
        tensor['shape'][-1] = old_axissize // new_itemsize
        tensor['dtype'] = dtype
        return hdr
    return block_view(block, header_transform)

def split_axis(block, axis, n, label=None):
    # Set function attributes to enable capture in nested function (closure)
    def header_transform(hdr, axis=axis, n=n, label=label):
        tensor = hdr['_tensor']
        if isinstance(axis, basestring):
            axis = tensor['labels'].index(axis)
        shape = tensor['shape']
        if shape[axis] == -1:
            # Axis is frame axis
            # TODO: Should assert even division here instead?
            # ***TODO: Why does pipeline deadlock when this doesn't divide?
            hdr['gulp_nframe'] = (hdr['gulp_nframe'] - 1) / n + 1
        else:
            # Axis is not frame axis
            if shape[axis] % n:
                raise ValueError("Split does not evenly divide axis (%i // %i)" %
                                 (tensor['shape'][axis], n))
            shape[axis] //= n
        shape.insert(axis + 1, n)
        if 'units' in tensor:
            tensor['units'].insert(axis + 1, tensor['units'][axis])
        if 'labels' in tensor:
            if label is None:
                label = tensor['labels'][axis] + "_split"
            tensor['labels'].insert(axis + 1, label)
        if 'scales' in tensor:
            tensor['scales'].insert(axis + 1, [0, tensor['scales'][axis][1]])
            tensor['scales'][axis][1] *= n
        return hdr
    return block_view(block, header_transform)

def merge_axes(block, axis1, axis2, label=None):
    def header_transform(hdr, axis1=axis1, axis2=axis2, label=label):
        tensor = hdr['_tensor']
        if isinstance(axis1, basestring):
            axis1 = tensor['labels'].index(axis1)
        if isinstance(axis2, basestring):
            axis2 = tensor['labels'].index(axis2)
        axis1, axis2 = sorted([axis1, axis2])
        if axis2 != axis1 + 1:
            raise ValueError("Merge axes must be adjacent")
        n = tensor['shape'][axis2]
        if n == -1:
            # Axis2 is frame axis
            raise ValueError("Second merge axis cannot be frame axis")
        elif tensor['shape'][axis1] == -1:
            # Axis1 is frame axis
            hdr['gulp_nframe'] *= n
        else:
            # Neither axis is frame axis
            tensor['shape'][axis1] *= n
        del tensor['shape'][axis2]
        if 'scales' in tensor and 'units' in tensor:
            scale1 = tensor['scales'][axis1][1]
            scale2 = tensor['scales'][axis2][1]
            units1 = tensor['units'][axis1]
            units2 = tensor['units'][axis2]
            scale2 = convert_units(scale2, units2, units1)
            if not isclose(scale1, n * scale2):
                raise ValueError("Scales of merge axes do not line up: "
                                 "%f != %f" % (scale1, n * scale2))
            tensor['scales'][axis1][1] = scale2
            del tensor['scales'][axis2]
            del tensor['units'][axis2]
        if 'labels' in tensor:
            if label is not None:
                tensor['labels'][axis1] = label
            del tensor['labels'][axis2]
        return hdr
    return block_view(block, header_transform)
