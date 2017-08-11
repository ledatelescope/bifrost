
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

# This is a minimal self-contained wrapper for the PortAudio library
# It supports basic reading and writing of audio streams

# Ubuntu 16.04:
#     sudo apt-get install portaudio19-dev

import ctypes
import atexit
from threading import Lock
import os

# Note: portaudio is MIT licensed
_lib = ctypes.cdll.LoadLibrary('libportaudio.so')

PaStream_ptr   = ctypes.c_void_p
PaDeviceIndex  = ctypes.c_int
PaSampleFormat = ctypes.c_ulong
PaTime         = ctypes.c_double
PaStreamFlags  = ctypes.c_ulong
PaHostApiIndex = ctypes.c_int

paNoError = 0

paNoFlag         = 0x0
paClipOff        = 0x1
paDitherOff      = 0x2
paNeverDropInput = 0x4

paUInt8 = 0x20
paInt8  = 0x10
paInt16 = 0x08
paInt24 = 0x04
paInt32 = 0x02

class PaStreamParameters(ctypes.Structure):
    _fields_ = [('device',                    PaDeviceIndex),
                ('channelCount',              ctypes.c_int),
                ('sampleFormat',              PaSampleFormat),
                ('suggestedLatency',          PaTime),
                ('hostApiSpecificStreamInfo', ctypes.c_void_p)]
    def __init__(self, device, channelCount, sampleFormat, suggestedLatency):
        self.device           = device
        self.channelCount     = channelCount
        self.sampleFormat     = sampleFormat
        self.suggestedLatency = suggestedLatency
        self.hostApiSpecificStreamInfo = ctypes.c_void_p(0)

class PaDeviceInfo(ctypes.Structure):
    _fields_ = [('structVersion',            ctypes.c_int),
                ('name',                     ctypes.c_char_p),
                ('hostApi',                  PaHostApiIndex),
                ('maxInputChannels',         ctypes.c_int),
                ('maxOutputChannels',        ctypes.c_int),
                ('defaultLowInputLatency',   PaTime),
                ('defaultLowOutputLatency',  PaTime),
                ('defaultHighInputLatency',  PaTime),
                ('defaultHighOutputLatency', PaTime),
                ('defaultSampleRate',        ctypes.c_double)]

_lib.Pa_GetDeviceCount.restype = PaDeviceIndex
_lib.Pa_GetDeviceInfo.restype  = ctypes.POINTER(PaDeviceInfo)
_lib.Pa_GetErrorText.restype   = ctypes.c_char_p
_lib.Pa_GetStreamTime.restype  = PaTime
_lib.Pa_OpenStream.argtypes = [ctypes.POINTER(PaStream_ptr),
                               ctypes.POINTER(PaStreamParameters),
                               ctypes.POINTER(PaStreamParameters),
                               ctypes.c_double,
                               ctypes.c_ulong,
                               PaStreamFlags,
                               ctypes.c_void_p,
                               ctypes.c_void_p]
_lib.Pa_CloseStream.argtypes = [PaStream_ptr]
_lib.Pa_StartStream.argtypes = [PaStream_ptr]
_lib.Pa_StopStream.argtypes  = [PaStream_ptr]
_lib.Pa_ReadStream.argtypes  = [PaStream_ptr,
                                ctypes.c_void_p,
                                ctypes.c_ulong]
_lib.Pa_WriteStream.argtypes = [PaStream_ptr,
                                ctypes.c_void_p,
                                ctypes.c_ulong]

class PortAudioError(RuntimeError):
    def __init__(self, msg):
        super(PortAudioError, self).__init__(msg)

def _check(err):
    if err != paNoError:
        raise PortAudioError(_lib.Pa_GetErrorText(err))

class suppress_fd(object):
    def __init__(self, fd):
        if   fd.lower() == 'stdout': fd = 1
        elif fd.lower() == 'stderr': fd = 2
        else: assert(isinstance(fd, int))
        self.fd = fd
        self.devnull = os.open(os.devnull, os.O_RDWR)
        self.stderr = os.dup(self.fd) # Save original
    def __enter__(self):
        os.dup2(self.devnull, self.fd) # Set stderr to devnull
    def __exit__(self, type, value, tb):
        os.dup2(self.stderr, self.fd) # Restore original
        os.close(self.devnull)

# Module-wide initialization/cleanup
with suppress_fd('stderr'):
    _check(_lib.Pa_Initialize())
atexit.register(_lib.Pa_Terminate)

class Stream(object):
    def __init__(self,
                 mode='r',
                 rate=44100,
                 channels=2,
                 nbits=16,
                 frames_per_buffer=1024,
                 input_device=None,
                 output_device=None):
        self.mode     = mode
        self.rate     = rate
        self.channels = channels
        self.nbits    = nbits
        self.frames_per_buffer = frames_per_buffer
        self.input_device  = input_device
        self.output_device = output_device
        # TODO: Should also allow paUInt8
        if   nbits ==  8: format = paInt8
        elif nbits == 16: format = paInt16
        elif nbits == 24: format = paInt24
        elif nbits == 32: format = paInt32
        else: raise ValueError("Invalid nbit")
        self.format = format
        if input_device is None:
            input_device = _lib.Pa_GetDefaultInputDevice()
        if output_device is None:
            output_device = _lib.Pa_GetDefaultOutputDevice()
        self.frame_nbyte = nbits // 8 * channels
        self.stream = PaStream_ptr()
        self.lock = Lock()
        stream_flags = PaStreamFlags(paClipOff)
        ilatency = _lib.Pa_GetDeviceInfo( input_device).contents.defaultLowInputLatency
        olatency = _lib.Pa_GetDeviceInfo(output_device).contents.defaultLowOutputLatency
        iparams = PaStreamParameters( input_device, channels, format, ilatency)
        oparams = PaStreamParameters(output_device, channels, format, olatency)
        use_input  = 'r' in mode or '+' in mode
        use_output = 'w' in mode or '+' in mode
        iparams_ptr = ctypes.pointer(iparams) if use_input  else None
        oparams_ptr = ctypes.pointer(oparams) if use_output else None
        # TODO: Consider adding a check with this:
        #PaError Pa_IsFormatSupported (const PaStreamParameters *inputParameters, const PaStreamParameters *outputParameters, double sampleRate)
        _check(_lib.Pa_OpenStream(ctypes.byref(self.stream),
                                  iparams_ptr,
                                  oparams_ptr,
                                  rate,
                                  frames_per_buffer,
                                  stream_flags,
                                  None,
                                  None))
        self.start()
    def close(self):
        self.stop()
        with self.lock:
            _check(_lib.Pa_CloseStream(self.stream))
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.close()
    def start(self):
        with self.lock:
            _check(_lib.Pa_StartStream(self.stream))
            self.running = True
    def stop(self):
        with self.lock:
            if self.running:
                _check(_lib.Pa_StopStream(self.stream))
                self.running = False
    def read(self, nframe):
        nbyte = nframe * self.frame_nbyte
        buf = ctypes.create_string_buffer("UNINITIALIZED"[:nbyte], nbyte)
        return self.readinto(buf)
    def readinto(self, buf):
        with self.lock:
            assert(len(buf) % self.frame_nbyte == 0)
            nframe = len(buf) // self.frame_nbyte
            # Note: This allows buf to be a buffer/memoryview object
            #         E.g., numpy.ndarray.data
            buf_view = (ctypes.c_byte * len(buf)).from_buffer(buf)
            # TODO: This returns PaInputOverflowed if input data were dropped
            #         by PortAudio since the last call (equivalent to dropping
            #         packets).
            _check(_lib.Pa_ReadStream(self.stream, buf_view, nframe))
            return buf
    def write(self, buf):
        with self.lock:
            assert(len(buf) % self.frame_nbyte == 0)
            nframe = len(buf) // self.frame_nbyte
            buf_view = (ctypes.c_byte * len(buf)).from_buffer(buf)
            _check(_lib.Pa_WriteStream(self.stream, buf_view, nframe))
            return buf
    def time(self):
        with self.lock:
            return _lib.Pa_GetStreamTime(self.stream)

def open(*args, **kwargs):
    return Stream(*args, **kwargs)

def get_device_count():
    return _lib.Pa_GetDeviceCount()

if __name__ == "__main__":
    import portaudio as audio
    import numpy as np
    print "Found %i audio devices" % audio.get_device_count()
    with audio.open(nbits=16) as audio_stream:
        nframe = 20
        print repr(audio_stream.read(nframe).raw)
        buf = -1 * np.ones(shape=[nframe, audio_stream.channels],
                           dtype=np.int16)
        audio_stream.readinto(buf.data)
        print buf
