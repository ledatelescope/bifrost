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
    