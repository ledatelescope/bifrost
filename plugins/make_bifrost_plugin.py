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

from ctypesgen import main as ctypeswrap


# Default Bifrost source path
BIFROST_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Makefile template
_MAKEFILE_PATH = os.path.dirname(os.path.abspath(__file__))
_MAKEFILE_TEMPLATE = open(os.path.join(_MAKEFILE_PATH, 'Makefile.template'), 'r').read()


def resolve_bifrost(bifrost_path=None):
    """
    Given a base path for a Bifrost installation, find all of the necessary 
    components for the Makefile.  Returns a four-element tuple of the
    configuration path, includes path, library path, and plugin scripts path.
    """
    
    # Get the Bifrost source path, if needed
    if bifrost_path is None:
        bifrost_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    # Setup the dependant paths
    ## Includes
    bifrost_include_path = bifrost_path+'/include'
    if not os.path.exists(os.path.join(bifrost_include_path, 'bifrost', 'ring.h')):
        ### Fallback to this being in the src directory
        bifrost_include_path = os.path.join(bifrost_path, 'src')
    ## Libraries
    bifrost_library_path = bifrost_path+'/lib'
    ## Plugin scripts
    bifrost_script_path = os.path.dirname(os.path.abspath(__file__))
    
    # Done
    return bifrost_include_path, bifrost_library_path, bifrost_script_path


def get_makefile_name(libname):
    """
    Given a library name, return the corresponding Makefile name.
    """
    
    return "Makefile.%s" % libname


def create_makefile(libname, includes, bifrost_path=None):
    """
    Given a library name, a set of header files, and the path to Bifrost,
    create a Makefile for building a Bifrost plugin and return the filename.
    """
    
    # Make sure the includes end up as a string
    if isinstance(includes, (list, tuple)):
        includes = " ".join(includes)
        
    # Get the Bifrost paths
    bifrost_include, bifrost_library, bifrost_script = resolve_bifrost(bifrost_path=bifrost_path)
       
    # Fill the template, save it, and return the filename
    template = _MAKEFILE_TEMPLATE.format(libname=libname,
                                         includes=includes,
                                         bifrost_include=bifrost_include,
                                         bifrost_library=bifrost_library,
                                         bifrost_script=bifrost_script)
    template = template.replace('-L. -lcufft_static_pruned', '')
    filename = get_makefile_name(libname)
    with open(filename, 'w') as fh:
        fh.write(template)
    return filename


def build(filename):
    """
    Given a Makefile name, run "make all".  Return True if successful, 
    False otherwise.
    """
    
    if not os.path.exists(filename):
        raise OSError("File '%s' does not exist" % filename)
    status = os.system("make -f %s all" % filename)
    return True if status == 0 else False


def clean(filename):
    """
    Given a Makefile name, run "make clean".  Return True if successful, 
    False otherwise.
    """
    
    if not os.path.exists(filename):
        raise OSError("File '%s' does not exist" % filename)
    status = os.system("make -f %s clean" % filename)
    return True if status == 0 else False


def purge(filename):
    """
    Given a a Makefile name, run "make clean" and then delete the Makefile.
    Return True if successful, False otherwise.
    """
    
    if not os.path.exists(filename):
        raise OSError("File '%s' does not exist" % filename)
    status = clean(filename)
    if status:
        try:
            os.unlink(filename)
        except OSError:
            status = False
    return status


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
    
