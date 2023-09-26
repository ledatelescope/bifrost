#!/usr/bin/env python3

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

import os
import sys
import glob
import argparse

from ctypesgen import main as ctypeswrap


# Default Bifrost source path
BIFROST_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Makefile template
_MAKEFILE_TEMPLATE = """
{preamble}

CXXFLAGS  += -I{bifrost_include} -I.
NVCCFLAGS += -I{bifrost_include} -I. -Xcompiler "-fPIC" $(NVCC_GENCODE)
LDFLAGS += -L{bifrost_library} -lbifrost

PYTHON_BINDINGS_FILE={libname}_generated.py
PYTHON_WRAPPER_FILE={libname}.py

.PHONY: all
all: lib{libname}.so $(PYTHON_BINDINGS_FILE) $(PYTHON_WRAPPER_FILE)

define run_ctypesgen
	{python} -c 'from ctypesgen import main as ctypeswrap; ctypeswrap.main()' -l$1 -I. -I{bifrost_include} $^ -o $@
	# WAR for 'const char**' being generated as POINTER(POINTER(c_char)) instead of POINTER(c_char_p)
	sed -i 's/POINTER(c_char)/c_char_p/g' $@
	# WAR for a buggy WAR in ctypesgen that breaks type checking and auto-byref functionality
	sed -i 's/def POINTER/def POINTER_not_used/' $@
	# WAR for a buggy WAR in ctypesgen that breaks string buffer arguments (e.g., as in address.py)
	sed -i 's/class String/String = c_char_p\\nclass String_not_used/' $@
	sed -i 's/String.from_param/String_not_used.from_param/g' $@
	sed -i 's/def ReturnString/ReturnString = c_char_p\\ndef ReturnString_not_used/' $@
	sed -i '/errcheck = ReturnString/s/^/#/' $@
endef

define run_wrapper
	{python} {bifrost_script}/wrap_bifrost_plugin.py $1
endef

lib{libname}.so: {libname}.o
	$(CXX) -o lib{libname}.so {libname}.o -lm -shared -fopenmp $(LDFLAGS)

%.o: %.cpp {includes}
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(GCCFLAGS) $(TARGET_ARCH) -c $(OUTPUT_OPTION) $<

%.o: %.cu {includes}
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler "$(GCCFLAGS)" $(TARGET_ARCH) -c $(OUTPUT_OPTION) $<

$(PYTHON_BINDINGS_FILE): {includes}
	$(call run_ctypesgen,{libname},{includes})

$(PYTHON_WRAPPER_FILE): $(PYTHON_BINDINGS_FILE)
	$(call run_wrapper,$(PYTHON_BINDINGS_FILE))

clean:
	rm -f lib{libname}.so {libname}.o $(PYTHON_BINDINGS_FILE) $(PYTHON_WRAPPER_FILE)

"""


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
    ## Configuration files
    bifrost_config_path = bifrost_path+'/include/bifrost/config'
    if not os.path.exists(os.path.join(bifrost_config_path, 'config.mk')):
        ### Fallback to this being in the directory itself
        bifrost_config_path = bifrost_path
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
    return bifrost_config_path, bifrost_include_path, bifrost_library_path, bifrost_script_path


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
    bifrost_config, bifrost_include, bifrost_library, bifrost_script = resolve_bifrost(bifrost_path=bifrost_path)
    
    preamble = ''
    with open(os.path.join(bifrost_include, 'Makefile'), 'r') as ch:
        in_block = 0
        block = ''
        dump_block = False
        for line in ch:
            if line.startswith('ifeq') or line.startswith('ifdef') or line.startswith('ifndef'):
                in_block += 1
            if in_block:
                block += line
                if line.startswith('endif'):
                    in_block -= 1
                    
                    if in_block == 0:
                        if (block.find('GPU') != -1 or block.find('NVCC') != -1 or block.find('FLAGS') != -1) \
                            and block.find('CUFFT') == -1:
                            preamble += block
                        block = ''
                        
            elif line.startswith('include') and line.find('autodep') == -1:
                preamble += line.replace('../', bifrost_config+'/')
            elif line.find('?=') != -1 and line.find('JIT') == -1 and line.find('WLFLAGS') == -1:
                preamble += line 
            elif len(line) < 3:
                preamble += line
    preamble = preamble.replace('-lcufft_static_pruned', '')
    
    # Sort out the old vs. new build system
    if os.path.exists(os.path.join(bifrost_include, 'bifrost', 'config.h')):
        preamble += 'CPPFLAGS += -DBF_HAVE_CONFIG_H=1\n'
        preamble += 'NVCCFLAGS += -DBF_HAVE_CONFIG_H=1\n'
        
    # Fill the template, save it, and return the filename
    template = _MAKEFILE_TEMPLATE.format(libname=libname,
                                         includes=includes,
                                         bifrost_config=bifrost_config,
                                         bifrost_include=bifrost_include,
                                         bifrost_library=bifrost_library,
                                         bifrost_script=bifrost_script,
                                         preamble=preamble,
                                         python=sys.executable)
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
    
