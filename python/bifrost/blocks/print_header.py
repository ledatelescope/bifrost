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
import pprint

from bifrost.pipeline import SinkBlock

from threading import Lock

class PrintHeaderBlock(SinkBlock):
    lock = Lock()
    def __init__(self, iring, *args, **kwargs):
        """Prints out the header of each new sequence of a ring

        Called by :meth:`bifrost.blocks.print_header`
        """
        super(PrintHeaderBlock, self).__init__(iring, *args, **kwargs)
    def on_sequence(self, iseq):
        ihdr = iseq.header
        with PrintHeaderBlock.lock:
            print("-----")
            print("Block", self.iring.owner.name, ihdr['name'])
            pprint.pprint(ihdr)
            print("-----")
    def on_sequence_end(self, iseq):
        pass
    def on_data(self, ispan):
        pass

def print_header(iring, *args, **kwargs):
    """Prints out the header of each new sequence of a ring.

    Use this for testing purposes to have a quick look
    at the contents of a ring when you are unsure. This
    is done using the simple python `print` statement,
    without any modification to the dictionary.

    Attributes
    ----------
    iring : Block
        A derivative of a Block object.
    *args
        Arguments to `bifrost.pipeline.TransformBlock`.
    **kwargs
        Keyword Arguments to `bifrost.pipeline.TransformBlock`.

    Returns
    -------
    `PrintHeaderBlock`
    """
    return PrintHeaderBlock(iring, *args, **kwargs)
