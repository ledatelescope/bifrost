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

from .copy import copy, CopyBlock
from .transpose import transpose, TransposeBlock
from .reverse import reverse, ReverseBlock
from .fft import fft, FftBlock
from .fftshift import fftshift, FftShiftBlock
from .fdmt import fdmt, FdmtBlock
from .detect import detect, DetectBlock
from .guppi_raw import read_guppi_raw, GuppiRawSourceBlock
from .print_header import print_header, PrintHeaderBlock
from .sigproc import read_sigproc, SigprocSourceBlock
from .sigproc import write_sigproc, SigprocSinkBlock
from .scrunch import scrunch, ScrunchBlock
from .accumulate import accumulate, AccumulateBlock
from .binary_io import BinaryFileReadBlock, BinaryFileWriteBlock
from .binary_io import binary_read, binary_write
from .unpack import unpack, UnpackBlock
from .quantize import quantize, QuantizeBlock
from .wav import read_wav, WavSourceBlock
from .wav import write_wav, WavSinkBlock
from .serialize import serialize, SerializeBlock, deserialize, DeserializeBlock
from .reduce import reduce, ReduceBlock
from .correlate import correlate, CorrelateBlock
from .convert_visibilities import convert_visibilities, ConvertVisibilitiesBlock

try: # Avoid error if portaudio library not installed
    from .audio import read_audio, AudioSourceBlock
except:
    pass

try: # Avoid error if psrdada library not installed
    from .psrdada import read_psrdada_buffer, PsrDadaSourceBlock
except:
    pass
