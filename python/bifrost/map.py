
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

from libbifrost import _bf, _check, _get, _array
import bifrost as bf
import numpy as np
import ctypes

def _is_literal(x):
    return isinstance(x, (int, long, float, complex))

def _convert_to_array(arg):
    if _is_literal(arg):
        arr = np.array(arg)
        if isinstance(arg, (int, long)) and -(1 << 31) <= arg < (1 << 31):
            arr = arr.astype(np.int32)
        # TODO: Any way to decide when these should be double-precision?
        elif isinstance(arg, float):
            arr = arr.astype(np.float32)
        elif isinstance(arg, complex):
            arr = arr.astype(np.complex64)
        arr.flags['WRITEABLE'] = False
        arg = arr
    return bf.asarray(arg)

def map(func_string, data, axis_names=None, shape=None,
        block_shape=None, block_axes=None):
    """Apply a function to a set of ndarrays.

    Args:
      func_string (str): The function to apply to the arrays, as a string (see
                   below for examples).
      data (dict): Map of string names to ndarrays or scalars.
      axis_names (list): List of string names by which each axis is referenced
                   in func_string.
      shape:       The shape of the computation. If None, the broadcast shape
                   of all data arrays is used.
      block_shape: The 2D shape of the thread block (y,x) with which the kernel
                   is launched.
                   This is a performance tuning parameter.
                   If NULL, a heuristic is used to select the block shape.
                   Changes to this parameter do _not_ require re-compilation of
                   the kernel.
      block_axes:  List of axis indices (or names) specifying the 2 computation
                   axes to which the thread block (y,x) is mapped.
                   This is a performance tuning parameter.
                   If NULL, a heuristic is used to select the block axes.
                   Values may be negative for reverse indexing.
                   Changes to this parameter _do_ require re-compilation of the
                   kernel.

    Note:
        Only GPU computation is currently supported.

    Examples::

      # Add two arrays together
      bf.map("c = a + b", {'c': c, 'a': a, 'b': b})

      # Compute outer product of two arrays
      bf.map("c(i,j) = a(i) * b(j)",
             {'c': c, 'a': a, 'b': b},
             axis_names=('i','j'))

      # Split the components of a complex array
      bf.map("a = c.real; b = c.imag", {'c': c, 'a': a, 'b': b})

      # Raise an array to a scalar power
      bf.map("c = pow(a, p)", {'c': c, 'a': a, 'p': 2.0})

      # Slice an array with a scalar index
      bf.map("c(i) = a(i,k)", {'c': c, 'a': a, 'k': 7}, ['i'], shape=c.shape)
    """
    narg = len(data)
    ndim = len(shape) if shape is not None else 0
    arg_arrays = []
    args = []
    arg_names = []
    if block_axes is not None:
        # Allow referencing axes by name
        block_axes = [axis_names.index(bax) if isinstance(bax, basestring)
                      else bax
                      for bax in block_axes]
    if block_axes is not None and len(block_axes) != 2:
        raise ValueError("block_axes must contain exactly 2 entries")
    if block_shape is not None and len(block_shape) != 2:
        raise ValueError("block_shape must contain exactly 2 entries")
    for key, arg in data.items():
        arg = _convert_to_array(arg)
        # Note: We must keep a reference to each array lest they be garbage
        #         collected before their corresponding BFarray is used.
        arg_arrays.append(arg)
        args.append(arg.as_BFarray())
        arg_names.append(key)
    _check(_bf.bfMap(ndim, _array(shape, dtype=ctypes.c_long),
                     _array(axis_names),
                     narg, _array(args), _array(arg_names),
                     func_string, _array(block_shape), _array(block_axes)))
