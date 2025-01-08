
# Copyright (c) 2016-2023, The Bifrost Authors. All rights reserved.
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

from bifrost.libbifrost import _bf, _check, _array
from bifrost.ndarray import asarray
from bifrost.ndarray import ndarray
import numpy as np
import ctypes
import glob
import os
from typing import Any, Dict, List, Optional
from bifrost.libbifrost_generated import BF_MAP_KERNEL_DISK_CACHE

from bifrost import telemetry
telemetry.track_module()

def _is_literal(x: Any) -> bool:
    return isinstance(x, (int, float, complex))

def _convert_to_array(arg: Any) -> ndarray:
    if _is_literal(arg):
        arr = np.array(arg)
        if isinstance(arg, int) and -(1 << 31) <= arg < (1 << 31):
            arr = arr.astype(np.int32)
        # TODO: Any way to decide when these should be double-precision?
        elif isinstance(arg, float):
            arr = arr.astype(np.float32)
        elif isinstance(arg, complex):
            arr = arr.astype(np.complex64)
        arr.flags['WRITEABLE'] = False
        arg = arr
    return asarray(arg)

def map(func_string: str, data: Dict[str,Any],
        axis_names: Optional[List[str]]=None,
        shape: Optional[List[int]]=None,
        func_name: Optional[str]=None,
        extra_code: Optional[str]=None,
        block_shape: Optional[List[int]]=None,
        block_axes: Optional[List[int]]=None) -> ndarray:
    """Apply a function to a set of ndarrays.

    Args:
      func_string (str): The function to apply to the arrays, as a string (see
                   below for examples).
      data (dict): Map of string names to ndarrays or scalars.
      axis_names (list): List of string names by which each axis is referenced
                   in func_string.
      shape:       The shape of the computation. If None, the broadcast shape
                   of all data arrays is used.
      func_name (str): Name of the function, for debugging purposes.
      extra_code (str): Additional code to be included at global scope.
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
    func_string = func_string.encode()
    if func_name is not None:
        func_name = func_name.encode()
    if extra_code is not None:
        extra_code = extra_code.encode()
    narg = len(data)
    ndim = len(shape) if shape is not None else 0
    arg_arrays = []
    args = []
    arg_names = []
    if block_axes is not None:
        # Allow referencing axes by name
        block_axes = [axis_names.index(bax) if isinstance(bax, str)
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
                     func_name, func_string, extra_code,
                     _array(block_shape), _array(block_axes)))

def list_map_cache() -> None:
    output = "Cache enabled: %s" % ('yes' if BF_MAP_KERNEL_DISK_CACHE else 'no')
    if BF_MAP_KERNEL_DISK_CACHE:
        cache_path = os.path.join(os.path.expanduser('~'), '.bifrost',
                                  _bf.BF_MAP_KERNEL_DISK_CACHE_SUBDIR)
        try:
            with open(os.path.join(cache_path, _bf.BF_MAP_KERNEL_DISK_CACHE_VERSION_FILE), 'r') as fh:
                version = fh.read()
            mapcache, runtime, driver = version.split(None, 2)
            mapcache = int(mapcache, 10)
            mapcache = f"{mapcache//1000}.{(mapcache//10) % 1000}"
            runtime = int(runtime, 10)
            runtime = f"{runtime//1000}.{(runtime//10) % 1000}"
            driver = int(driver, 10)
            driver = f"{driver//1000}.{(driver//10) % 1000}"
            
            entries = glob.glob(os.path.join(cache_path, '*.inf'))
            
            output += f"\nCache version: {mapcache} (map cache) {runtime} (runtime), {driver} (driver)"
            output += f"\nCache entries: {len(entries)}"
        except OSError:
            pass
            
    print(output)


def clear_map_cache() -> None:
    _check(_bf.bfMapClearCache())
