
# Copyright (c) 2016-2022, The Bifrost Authors. All rights reserved.
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

# Python2 compatibility
from __future__ import absolute_import

from ctypes import c_ulong, pointer as c_pointer
from bifrost.libbifrost import _bf, _check, _get, BifrostObject

from bifrost import telemetry
telemetry.track_module()

def set_device(device):
    if isinstance(device, int):
        _check(_bf.bfDeviceSet(device))
    else:
        try:
            device = device.encode()
        except AttributeError:
            # Python2 catch
            pass
        _check(_bf.bfDeviceSetById(device))

def get_device():
    return _get(_bf.bfDeviceGet)

def set_stream(stream):
    """Set the CUDA stream to the provided stream handle"""
    stream = c_ulong(stream)
    _check(_bf.bfStreamSet(c_pointer(stream)))
    return True
    
def get_stream():
    """Get the current CUDA stream and return its address"""
    stream = c_ulong(0)
    _check(_bf.bfStreamGet(c_pointer(stream)))
    return stream.value

class ExternalStream(object):
    """Context manager to use a stream created outside Bifrost"""
    def __init__(self, stream):
        self._stream = stream
    def __del__(self):
        try:
            set_stream(self._orig_stream)
        except AttributeError:
            pass
    def use(self):
        """Make the external stream the default stream.  The original Bifrost
        stream will be restored when this object is deleted.
        
        To temporirly switch streams use the 'with' statement."""
        self._orig_stream = get_stream()
        # cupy stream?
        stream = getattr(self._stream, 'ptr', None)
        if stream is None:
            # pycuda stream?
            stream = getattr(self._stream, 'handle', None)
        if stream is None:
            stream = self._stream
        set_stream(stream)
    def __enter__(self):
        self.use()
        return self
    def __exit__(self, type, value, tb):
        set_stream(self._orig_stream)
        del self._orig_stream

def stream_synchronize():
    _check(_bf.bfStreamSynchronize())

def set_devices_no_spin_cpu():
    """Sets a flag on all GPU devices that tells them not to spin the CPU when
    synchronizing. This is useful for reducing CPU load in GPU pipelines.

    This function must be called _before_ any GPU devices are
    initialized (i.e., at the start of the process)."""
    _check(_bf.bfDevicesSetNoSpinCPU())

class Graph(BifrostObject):
    """Context manager to use create a use a CUDA graph inside Bifrost"""
    def __init__(self):
        BifrostObject.__init__(self, _bf.bfGraphCreate, _bf.bfGraphDestroy)
        _check( _bf.bfGraphInit(self.obj) )
        self._init_pass = True
    @property
    def created(self):
        """Return whether or not the graph as been created"""
        return bool(_get(_bf.bfGraphCreated, self.obj))
    def __enter__(self):
        if not self.created:
            if not self._init_pass:
                _check( _bf.bfGraphBeginCapture(self.obj) )
                self._init_pass = False
    def __exit__(self, type, value, tb):
        if self.created:
            _check( _bf.bfGraphExecute(self.obj) )
        elif not self._init_pass:
            _check( _bf.bfGraphEndCapture(self.obj) )
    def copy_array(self, dst, src):
        """Version of bifrost.ndarray.copy_array that *does not* call
        bifrost.device.stream_synchronize() under any circumstance"""
        _check(_bf.bfArrayCopy(dst.as_BFarray(),
                               src.as_BFarray()))
