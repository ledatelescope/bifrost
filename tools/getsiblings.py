#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017, The Bifrost Authors. All rights reserved.
# Copyright (c) 2017, The University of New Mexico. All rights reserved.
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
import getopt


def usage(exitCode=None):
    print """%s - Get sibling cores on HT systems

Usage: %s [OPTIONS] [core [core [...]]]

Options:
-h, --help                  Display this help information
""" % (os.path.basename(__file__), os.path.basename(__file__))

    if exitCode is not None:
        sys.exit(exitCode)
    else:
        return True


def parseOptions(args):
    config = {}
    # Command line flags - default values
    config['args'] = []

    # Read in and process the command line flags
    try:
        opts, args = getopt.getopt(args, "h", ["help",])
    except getopt.GetoptError, err:
        # Print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage(exitCode=2)

    # Work through opts
    for opt, value in opts:
        if opt in ('-h', '--help'):
            usage(exitCode=0)
        else:
            assert False

    # Add in arguments
    config['args'] = args

    # Return configuration
    return config


def _readSiblings():
    cpus = glob.glob('/sys/devices/system/cpu/cpu*/topology/thread_siblings_list')

    siblings = {}
    for cpu in cpus:
        cid = cpu.split('/topology', 1)[0]
        cid = cid.rsplit('cpu', 1)[1]
        cid = int(cid, 10)

        fh = open(cpu)
        data = fh.read()
        fh.close()

        siblings[cid] = [int(v,10) for v in data.split(',')]

    return siblings


def main(args):
    config = parseOptions(args)

    siblings = _readSiblings()

    if len(config['args']) == 0:
        for cpu in sorted(siblings.keys()):
            print "%i: %s" % (cpu, str(siblings[cpu]))
    else:
        for cpu in config['args']:
            cpu = int(cpu, 10)
            try:
                data = str(siblings[cpu])
            except KeyError:
                data = "not found"
            print "%i: %s" % (cpu, str(siblings[cpu]))


if __name__ == '__main__':
    main(sys.argv[1:])

