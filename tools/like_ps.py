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
import time
import getopt
import subprocess

from bifrost.proclog import load_by_pid


BIFROST_STATS_BASE_DIR = '/dev/shm/bifrost/'

def usage(exitCode=None):
    print """%s - Display details of running bifrost processes

Usage: %s [OPTIONS]

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


def _getProcessDetails(pid):
    """
    Use a call to 'ps' to get details about the specified PID.  These details
    include:
      * the user running the process, 
      * the CPU usage of the process, 
      * the memory usage of the process, 
      * the process elapsed time, and
      * the number of threads.
    These are returned as a dictionary.

    NOTE::  Using 'ps' to get this is slightly easier than directly querying 
          /proc, altough that method might be preferred.

    NOTE::  Many of these details could be avoided by using something like the
          Python 'psutil' module.
    """

    data = {'user':'', 'cpu':0.0, 'mem':0.0, 'etime':'00:00', 'threads':0}
    try:
        output = subprocess.check_output('ps o user,pcpu,pmem,etime,nlwp %i' % pid, shell=True)
        output = output.split('\n')[1]
        fields = output.split(None, 4)
        data['user'] = fields[0]
        data['cpu'] = float(fields[1])
        data['mem'] = float(fields[2])
        data['etime'] = fields[3].replace('-', 'd ')
        data['threads'] = int(fields[4], 10)
    except subprocess.CalledProcessError:
        pass
    return data


def _getCommandLine(pid):
    """
    Given a PID, use the /proc interface to get the full command line for 
    the process.  Return an empty string if the PID doesn't have an entry in
    /proc.
    """

    cmd = ''

    try:
        with open('/proc/%i/cmdline' % pid, 'r') as fh:
            cmd = fh.read()
            cmd = cmd.replace('\0', ' ')
            fh.close()
    except IOError:
        pass
    return cmd


def main(args):
    config = parseOptions(args)

    pidDirs = glob.glob(os.path.join(BIFROST_STATS_BASE_DIR, '*'))
    pidDirs.sort()

    for pidDir in pidDirs:
        pid = int(os.path.basename(pidDir), 10)
        contents = load_by_pid(pid)

        details = _getProcessDetails(pid)
        cmd = _getCommandLine(pid)

        if cmd == '' and details['user'] == '':
            continue

        print "PID: %i" % pid
        print "  Command: %s" % cmd
        print "  User: %s" % details['user']
        print "  CPU Usage: %.1f%%" % details['cpu']
        print "  Memory Usage: %.1f%%" % details['mem']
        print "  Elapsed Time: %s" % details['etime']
        print "  Thread Count: %i" % details['threads']
        print "  Rings:"
        rings = []
        for block in contents.keys():
            for log in contents[block].keys():
                if log not in ('in', 'out'):
                    continue
                for key in contents[block][log]:
                    if key[:4] == 'ring':
                        value = contents[block][log][key]
                        if value not in rings:
                            rings.append( value )
        for i,ring in enumerate(rings):
            print "    %i: %s" % (i, ring)
        print "  Blocks:"
        for block in contents.keys():
            rins, routs = [], []
            for log in contents[block].keys():
                if log not in ('in', 'out'):
                    continue
                for key in contents[block][log]:
                    if key[:4] == 'ring':
                        value = contents[block][log][key]
                        if log == 'in':
                            if value not in rins:
                                rins.append( value )
                        else:
                            if value not in routs:
                                routs.append( value )
            print "    %s" % block
            if len(rins) > 0:
                print "      -> read ring(s): %s" % (" ".join(["%i" % rings.index(v) for v in rins]),)
            if len(routs) > 0:
                print "      -> write ring(s): %s" % (" ".join(["%i" % rings.index(v) for v in routs]),)
            if len(contents[block].keys()) > 0:
                print "      -> log(s): %s" % (" ".join(contents[block].keys()),)


if __name__ == "__main__":
    main(sys.argv[1:])

