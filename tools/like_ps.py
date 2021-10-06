#!/usr/bin/env python

# Copyright (c) 2017-2021, The Bifrost Authors. All rights reserved.
# Copyright (c) 2017-2021, The University of New Mexico. All rights reserved.
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

# Python2 compatibility
from __future__ import print_function

import os
import glob
import argparse
import subprocess

os.environ['VMA_TRACELEVEL'] = '0'
from bifrost.proclog import load_by_pid

from bifrost import telemetry
telemetry.track_script()


BIFROST_STATS_BASE_DIR = '/dev/shm/bifrost/'


def get_process_details(pid):
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
        try:
            output = output.decode()
        except AttributeError:
            # Python2 catch
            pass
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


def get_command_line(pid):
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


def get_best_size(value):
    """
    Give a size in bytes, convert it into a nice, human-readable value 
    with units.
    """
    
    if value >= 1024.0**4:
        value = value / 1024.0**4
        unit = 'TB'
    elif value >= 1024.0**3:
        value = value / 1024.0**3
        unit = 'GB'
    elif value >= 1024.0**2:
        value = value / 1024.0**2
        unit = 'MB'
    elif value >= 1024.0:
        value = value / 1024.0
        unit = 'kB'
    else:
        unit = 'B'
    return value, unit


def main(args):
    pidDirs = glob.glob(os.path.join(BIFROST_STATS_BASE_DIR, '*'))
    pidDirs.sort()

    for pidDir in pidDirs:
        pid = int(os.path.basename(pidDir), 10)
        contents = load_by_pid(pid)

        details = get_process_details(pid)
        cmd = get_command_line(pid)

        if cmd == '' and details['user'] == '':
            continue

        print("PID: %i" % pid)
        print("  Command: %s" % cmd)
        print("  User: %s" % details['user'])
        print("  CPU Usage: %.1f%%" % details['cpu'])
        print("  Memory Usage: %.1f%%" % details['mem'])
        print("  Elapsed Time: %s" % details['etime'])
        print("  Thread Count: %i" % details['threads'])
        print("  Rings:")
        rings = []
        ring_details = {}
        for block in contents.keys():
            if block == 'rings':
                for ring in contents[block].keys():
                    ring_details[ring] = {}
                    for key in contents[block][ring]:
                        ring_details[ring][key] = contents[block][ring][key]
                continue
                
            for log in contents[block].keys():
                if log not in ('in', 'out'):
                    continue
                for key in contents[block][log]:
                    if key[:4] == 'ring':
                        value = contents[block][log][key]
                        if value not in rings:
                            rings.append( value )
        for i,ring in enumerate(rings):
            try:
                dtls = ring_details[ring]
                sz, un = get_best_size(dtls['stride']*dtls['nringlet'])
                print("    %i: %s on %s of size %.1f %s" % (i, ring, dtls['space'], sz, un))
            except KeyError:
                print("    %i: %s" % (i, ring))
        print("  Blocks:")
        for block in contents.keys():
            if block == 'rings':
                continue
                
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
            print("    %s" % block)
            if len(rins) > 0:
                print("      -> read ring(s): %s" % (" ".join(["%i" % rings.index(v) for v in rins]),))
            if len(routs) > 0:
                print("      -> write ring(s): %s" % (" ".join(["%i" % rings.index(v) for v in routs]),))
            if len(contents[block].keys()) > 0:
                print("      -> log(s): %s" % (" ".join(contents[block].keys()),))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Display details of running Bifrost pipelines',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    args = parser.parse_args()
    main(args)
    
