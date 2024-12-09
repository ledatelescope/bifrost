
# Copyright (c) 2019-2023, The Bifrost Authors. All rights reserved.
# Copyright (c) 2019-2023, The University of New Mexico. All rights reserved.
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
import os
import re
import sys
import glob

try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO

TEST_DIR = os.path.dirname(__file__)
TOOLS_DIR = os.path.join(TEST_DIR, '..', 'tools')
TESTBENCH_DIR = os.path.join(TEST_DIR, '..', 'testbench')

run_scripts_tests = False
try:
    from pylint.lint import Run
    from pylint.reporters.text import TextReporter
    run_scripts_tests = True
except ImportError:
    pass
run_scripts_tests &= (sys.version_info[0] >= 3)

_LINT_RE = re.compile('(?P<module>.*?):(?P<line>[0-9]+): \[(?P<type>.*?)\] (?P<info>.*)')

@unittest.skipUnless(run_scripts_tests, "requires the 'pylint' module")
class ScriptTest(unittest.TestCase):
    def _test_script(self, script):
        self.assertTrue(os.path.exists(script))
        
        pylint_output = StringIO()
        reporter = TextReporter(pylint_output)
        Run([script, '-E', '--extension-pkg-whitelist=numpy'], reporter=reporter, exit=False)
        out = pylint_output.getvalue()
        out_lines = out.split('\n')
        
        for line in out_lines:
            #if line.find("Module 'numpy") != -1:
            #    continue
            #if line.find("module 'scipy.fftpack") != -1:
            #    continue
                
            mtch = _LINT_RE.match(line)
            if mtch is not None:
                line_no, type, info = mtch.group('line'), mtch.group('type'), mtch.group('info')
                self.assertEqual(type, None, "%s:%s - %s" % (os.path.basename(script), line_no, info))
                
    def test_tools(self):
        scripts = glob.glob(os.path.join(TOOLS_DIR, '*.py'))
        for script in scripts:
            self._test_script(script)
    def test_testbench(self):
        scripts = glob.glob(os.path.join(TESTBENCH_DIR, '*.py'))
        for script in scripts:
            self._test_script(script)
            
