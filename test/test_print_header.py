
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

import unittest
import io

import bifrost.pipeline as bfp
import bifrost.blocks as blocks

from contextlib2 import redirect_stdout, ExitStack

class TestPrintHeader(unittest.TestCase):
    """Test all aspects of the print header block"""
    def setUp(self):
        self.fil_file = "./data/2chan4bitNoDM.fil"
    def test_read_sigproc(self):
        """Capture print output, assert it is a long string"""
        gulp_nframe = 101

        stdout = io.BytesIO()
        with ExitStack() as stack:
            pipeline = stack.enter_context(bfp.Pipeline())
            stack.enter_context(redirect_stdout(stdout))

            rawdata = blocks.sigproc.read_sigproc([self.fil_file], gulp_nframe)
            print_header_block = blocks.print_header(rawdata)
            pipeline.run()
        print_header_dump = stdout.getvalue()
        self.assertGreater(len(print_header_dump), 10)
