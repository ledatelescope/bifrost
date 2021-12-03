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

import glob
import argparse

from bifrost import telemetry
telemetry.track_script()


def read_siblings():
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
    siblings = read_siblings()

    if len(args.core) == 0:
        for cpu in sorted(siblings.keys()):
            print("%i: %s" % (cpu, str(siblings[cpu])))
    else:
        for cpu in args.core:
            try:
                data = str(siblings[cpu])
            except KeyError:
                data = "not found"
            print("%i: %s" % (cpu, str(siblings[cpu])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get sibling cores on hyper-threaded systems',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('core', type=int, nargs='*',
                        help='core to query')
    args = parser.parse_args()
    main(args)
    
