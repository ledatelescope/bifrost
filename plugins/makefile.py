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

"""
Module to help build the C++/CUDA side of a Bifrost plugin.
"""

import os


__all__ = ['get_makefile_name', 'create_makefile', 'build', 'clean', 'purge']


_MAKEFILE_TEMPLATE = os.path.join(os.path.dirname(__file__), 'makefile.tmpl')


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
        
    # Get the Bifrost source path, if needed
    if bifrost_path is None:
        bifrost_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    # Load in the template
    with open(_MAKEFILE_TEMPLATE, 'r') as fh:
        template = fh.read()
        
    # Fill the template, save it, and return the filename
    template = template.format(libname=libname,
                               includes=includes,
                               bifrost=bifrost_path)
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
    
