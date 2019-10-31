#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import glob

from makefile import *

from ctypesgen import main as ctypeswrap


BIFROST_PATH='/tmp/bifrost'


filename = sys.argv[1]
libname = os.path.basename(filename)
libname = os.path.splitext(libname)[0]
incname = glob.glob("%s*.h" % libname)
incname.extend(glob.glob("%s*.hpp" % libname))

makename = get_makefile_name(libname)

trigger_rebuild = False
if not os.path.exists(makename):
    trigger_rebuild = True
elif os.path.getmtime(makename) < os.path.getmtime(__file__):
    trigger_rebuild = True

# Part 1:  Build a Makefile
if trigger_rebuild:
    create_makefile(libname, incname, bifrost_path=BIFROST_PATH)

# Part 2:  Build and wrap
status = clean(makename)
if not status:
    sys.exit(1)
status = build(makename)
if not status:
    sys.exit(status)
    