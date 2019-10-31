#!/usr/bin/env python

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
    ext = os.path.splitext(filename)[1]
    if ext not in ('.cpp', '.cu'):
        raise RuntimeError("Unknown file extension '%s'" % ext)
        
    # Build up the set of names that we need to make progress
    libname = os.path.basename(filename)
    libname = os.path.splitext(libname)[0]
    incname = glob.glob("%s*.h" % libname)
    incname.extend(glob.glob("%s*.hpp" % libname))
    
    # Get the name of the Makefile for this plugin
    makename = get_makefile_name(libname)
    
    # Part 1:  Build the Makefile
    trigger_rebuild = args.force
    if not os.path.exists(makename):
        trigger_rebuild = True
    elif os.path.getmtime(makename) < os.path.getmtime(__file__) \
          or os.path.getmtime(makename) < os.path.getmtime('makefile.py') \
          or os.path.getmtime(makename) < os.path.getmtime('makefile.tmpl') \
          or os.path.getmtime(makename) < os.path.getmtime('wrap_plugin.py') \
          or os.path.getmtime(makename) < os.path.getmtime('wrap_plugin.tmpl'):
        trigger_rebuild = True
        
    if trigger_rebuild:
        create_makefile(libname, incname, bifrost_path=args.bifrost_path)
        
    # Part 2:  Build and wrap
    status = clean(makename)
    if not status:
        sys.exit(1)
    status = build(makename)
    if not status:
        sys.exit(status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Given a .cpp or .cu file that contains the source for a Bifrost plugin, create a Makefile, build the plugin, and create the Python wrappers needed to use the plugin.')
    parser.add_argument('filename', type=str,
                        help='filename to compile and wrap')
    parser.add_argument('-f', '--force', action='store_true',
                       help='force a rebuild of the Makefile')
    parser.add_argument('-b', '--bifrost-path', type=str, default=BIFROST_PATH,
                        help='path to the Bifrost source directory')
    args = parser.parse_args()
    main(args)
    