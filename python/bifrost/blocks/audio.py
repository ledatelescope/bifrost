
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

from bifrost.pipeline import SourceBlock
import bifrost.portaudio as audio

class AudioSourceBlock(SourceBlock):
    def create_reader(self, kwargs):
        self.reader = audio.open(mode='r', **kwargs)
        return self.reader
    def on_sequence(self, reader, kwargs):
        if 'frames_per_buffer' not in kwargs:
            kwargs['frames_per_buffer'] = self.gulp_nframe
        ohdr = {
            '_tensor': {
                'dtype':  'i' + str(reader.nbits),
                'shape':  [-1, reader.channels],
                'labels': ['time', 'pol'],
                'scales': [1. / reader.rate, None],
                'units':  ['s', None]
            },
            'frame_rate':   reader.rate,
            'input_device': reader.input_device,
            'name': str(id(reader))
        }
        return [ohdr]
    def on_data(self, reader, ospans):
        ospan = ospans[0]
        try:
            reader.readinto(ospan.data)
        except audio.PortAudioError:
            #raise StopIteration
            return [0]
        nframe = ospan.shape[0]
        return [nframe]
    def stop(self):
        self.reader.stop()

def read_audio(audio_kwargs, gulp_nframe, *args, **kwargs):
    """Read data from an audio input device.

    Requires the portaudio library to be installed::

      $ sudo apt-get install portaudio19-dev

    Args:
        audio_kwargs (list): List of dicts containing audio input parameters
           Defaults:

            ``rate=44100``

            ``channels=2``

            ``nbits=16``

            ``frames_per_buffer=1024``

            ``input_device=None``

        gulp_nframe (int):   No. frames to read at a time.
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.

    **Tensor semantics**::

        Output: ['time', 'pol'], dtype = i*, space = SYSTEM

    Returns:
        AudioBlock: A new block instance.
    """
    # Note: audio_kwargs used in place of sourcenames
    return AudioSourceBlock(audio_kwargs, gulp_nframe, *args, **kwargs)
