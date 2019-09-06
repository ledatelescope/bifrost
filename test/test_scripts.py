#!/usr/bin/env python

# Copyright (c) 2019, The Bifrost Authors. All rights reserved.
# Copyright (c) 2019, The University of New Mexico. All rights reserved.
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
import imp
import sys

currentDir = os.path.abspath(os.getcwd())
if os.path.exists(os.path.join(currentDir, 'download_test_data.sh')):
    modInfoBuild = imp.find_module('bifrost', [os.path.join(currentDir, '..', 'python')])
    MODULE_BUILD =  os.path.abspath(modInfoBuild[1])
elif os.path.exists(os.path.joni(currentDir, 'user.mk')):
    modInfoBuild = imp.find_module('bifrost', [os.path.join(currentDir, 'python')])
    MODULE_BUILD =  os.path.abspath(modInfoBuild[1])
else:
    MODULE_BUILD = None

run_scripts_tests = False
try:
    from pylint import epylint as lint
    if MODULE_BUILD is not None:
        run_scripts_tests = True
except ImportError:
    pass

_LINT_RE = re.compile('(?P<module>.*?)\:(?P<line>\d+)\: \[(?P<type>.*?)\] (?P<info>.*)')

@unittest.skipUnless(run_scripts_tests, "requires the 'pylint' module")
class ScriptTest(unittest.TestCase):
    def _test_script(self, filename):
        self.assertTrue(os.path.exists(filename))
        out, err = lint.py_run("%s -E --init-hook='import sys; sys.path=[%s]; sys.path.insert(0, \"%s\")'" % (filename, ",".join(['"%s"' % p for p in sys.path]), os.path.dirname(MODULE_BUILD)), return_std=True)
        out_lines = out.read().split('\n')
        err_lines = err.read().split('\n')
        out.close()
        err.close()
        
        for line in out_lines:
            print line
            if line.find("Module 'numpy") != -1:
                continue
                
            mtch = _LINT_RE.match(line)
            if mtch is not None:
                line_no, type, info = mtch.group('line'), mtch.group('type'), mtch.group('info')
                self.assertEqual(type, None, "%s:%s - %s" % (os.path.basename(filename), line_no, info))
    def test_getirq(self):
        self._test_script(os.path.join(MODULE_BUILD, '..', '..', 'tools', 'getirq.py'))
    def test_getsiblings(self):
        self._test_script(os.path.join(MODULE_BUILD, '..', '..', 'tools', 'getsiblings.py'))
    def test_like_bmon(self):
        self._test_script(os.path.join(MODULE_BUILD, '..', '..', 'tools', 'like_bmon.py'))
    def test_like_pmap(self):
        self._test_script(os.path.join(MODULE_BUILD, '..', '..', 'tools', 'like_pmap.py'))
    def test_like_ps(self):
        self._test_script(os.path.join(MODULE_BUILD, '..', '..', 'tools', 'like_ps.py'))
    def test_like_top(self):
        self._test_script(os.path.join(MODULE_BUILD, '..', '..', 'tools', 'like_top.py'))
    def test_pipeline2dot(self):
        self._test_script(os.path.join(MODULE_BUILD, '..', '..', 'tools', 'pipeline2dot.py'))
    def test_setirq(self):
        self._test_script(os.path.join(MODULE_BUILD, '..', '..', 'tools', 'setirq.py'))


