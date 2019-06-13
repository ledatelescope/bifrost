
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

from ctypes import c_ulong, byref
from libbifrost import _bf, _check, _get

def set_device(device):
    if isinstance(device, int):
        _check(_bf.bfDeviceSet(device))
    else:
        _check(_bf.bfDeviceSetById(device))

def get_device():
    return _get(_bf.bfDeviceGet)

def create_stream(nonblocking=False):
    """Create a new CUDA stream and return it as a ctypes.c_ulong instance.  
    If the `nonblocking` is True then the stream may run independently with
    respect to stream 0."""
    stream = c_ulong(0)
    _check(_bf.bfStreamCreate(byref(stream), nonblocking))
    return stream


def get_stream():
    """Get the current CUDA stream and return it as a ctypes.c_ulong instance."""
    stream = c_ulong(0)
    _check(_bf.bfStreamGet(byref(stream)))
    return stream

def set_stream(stream):
    """Set the CUDA stream to the provided ctypes.c_ulong instance."""
    if not isinstance(stream, c_ulong):
        raise TypeError("Expected a ctypes.u_int instance")
    _check(_bf.bfStreamSet(byref(stream)))
    return True

def stream_synchronize():
    _check(_bf.bfStreamSynchronize())

def set_devices_no_spin_cpu():
    """Sets a flag on all GPU devices that tells them not to spin the CPU when
    synchronizing. This is useful for reducing CPU load in GPU pipelines.

    This function must be called _before_ any GPU devices are
    initialized (i.e., at the start of the process)."""
    _check(_bf.bfDevicesSetNoSpinCPU())
