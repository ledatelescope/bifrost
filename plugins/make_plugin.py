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

from __future__ import print_function

import os
import sys
import glob
import argparse

from makefile import *

from ctypesgen import main as ctypeswrap


# Default Bifrost source path
BIFROST_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(args):
    filename = args.filename
    if os.path.dirname(os.path.abspath(filename)) \
       != os.path.abspath(os.getcwd()):
        raise RuntimeError("%s must be run from the same directory as %s" % (os.path.basename(__file__), os.path.basename(filename)))
    ext = os.path.splitext(filename)[1]
    if ext not in ('.cpp', '.cu'):
        raise RuntimeError("Unknown file extension '%s'" % ext)
        
    # Build up the set of names that we need to make progress
    libname = os.path.basename(filename)
    libname = os.path.splitext(libname)[0]
    incname = glob.glob("%s*.h" % libname)
    incname.extend(glob.glob("%s*.hpp" % libname))
    if libname+'.h' not in incname:
        raise RuntimeError("Cannot find the associated C header file: %s" % libname+'.h')
        
    # Get the name of the Makefile for this plugin
    makename = get_makefile_name(libname)
    
    # Part 1:  Build the Makefile
    create_makefile(libname, incname, bifrost_path=args.bifrost_path)
        
    # Part 2:  Build and wrap
    status = clean(makename)
    if not status:
        sys.exit(1)
    status = build(makename)
    if not status:
        sys.exit(status)
        
    # Part 3:  Clean up
    os.unlink(makename)
    objnames = glob.glob("%s*.o" % libname)
    for objname in objnames:
        os.unlink(objname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Given a .cpp or .cu file that contains the source for a Bifrost plugin, build the plugin and create the Python wrappers needed to use it.')
    parser.add_argument('filename', type=str,
                        help='filename to compile and wrap')
    parser.add_argument('-b', '--bifrost-path', type=str, default=BIFROST_PATH,
                        help='path to the Bifrost source directory')
    args = parser.parse_args()
    main(args)
    
