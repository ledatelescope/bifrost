# -*- coding: utf-8 -*-

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

from libbifrost import _bf, _check, _get, BifrostObject

import os
import time
import ctypes
import numpy as np

try:
    import simplejson as json
except ImportError:
    print "WARNING: Install simplejson for better performance"
    import json

class ProcLog(BifrostObject):
    def __init__(self, name):
        BifrostObject.__init__(
            self, _bf.bfProcLogCreate, _bf.bfProcLogDestroy, name)
    def update(self, contents):
        """Updates (replaces) the contents of the log
        contents: string or dict containing data to write to the log
        """
        if contents is None:
            raise ValueError("Contents cannot be None")
        if isinstance(contents, dict):
            contents = '\n'.join(['%s : %s' % item
                                  for item in contents.items()])
        _check(_bf.bfProcLogUpdate(self.obj, contents))

def _multi_convert(value):
    """
    Function try and convert numerical values to numerical types.
    """

    try:
        value = int(value, 10)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    return value

def load_by_filename(filename):
    """
    Function to read in a ProcLog file and return the contents as a 
    dictionary.
    """

    contents = {}
    with open(filename, 'r') as fh:
        ## Read the file all at once to avoid problems but only after it has a size
        for attempt in xrange(5):
            if os.path.getsize(filename) != 0:
                break
            time.sleep(0.001)
        lines = fh.read()

        ## Loop through lines
        for line in lines.split('\n'):
            ### Parse the key : value pairs
            try:
                key, value = line.split(':', 1)
            except ValueError:
                continue

            ### Trim off excess whitespace
            key = key.strip().rstrip()
            value = value.strip().rstrip()

            ### Convert and save
            contents[key] = _multi_convert(value)
            
    # Done
    return contents

def load_by_pid(pid, include_rings=False):
    """
    Function to read in and parse all ProcLog files associated with a given 
    process ID.  The contents of these files are returned as a collection of
    dictionaries ordered by:
      block name
        ProcLog name
           entry name
    """


    # Make sure we have a directory to load from
    baseDir = os.path.join('/dev/shm/bifrost/', str(pid))
    if not os.path.isdir(baseDir):
        raise RuntimeError("Cannot find log directory associated with PID %s" % pid)

    # Load
    contents = {}
    for parent,subnames,filenames in os.walk(baseDir):
        for filename in filenames:
            filename = os.path.join(parent, filename)

            ## Extract the block and logfile names
            logName = os.path.basename(filename)
            blockName = os.path.basename( os.path.dirname(filename) )
            if blockName == 'rings' and not include_rings:
                continue

            ## Load the file's contents
            try:
                subContents = load_by_filename(filename)
            except IOError:
                continue

            ## Save
            try:
                contents[blockName][logName] = subContents
            except KeyError:
                contents[blockName] = {logName:subContents}

    # Done
    return contents
