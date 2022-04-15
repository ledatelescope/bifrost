#!/usr/bin/env python

# Copyright (c) 2017-2021, The Bifrost Authors. All rights reserved.
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

import argparse

from bifrost import telemetry
telemetry.track_script()


def compute_mask(cpu):
    """
    Given a CPU number, return a bitmask that can be used in /proc/irq to set
    the processor affinity for an interrupt.
    """

    return 1<<cpu


def write_irq_smp_affinity(irq, mask):
    """
    Write the given process affinity mask to /proc/irq for the speicifed 
    interrupt.
    """

    mask_str = "%08x" % mask
    filename = "/proc/irq/%i/smp_affinity" % irq
    with open(filename, 'w') as f:
        f.write(mask_str+"\n")


def main(args):
    fh = open('/proc/interrupts', 'r')
    lines = fh.read()
    fh.close()

    irqs = {}
    for line in lines.split('\n'):
        if line.find(args.interface) != -1:
            fields = line.split()
            irq = int(fields[0][:-1], 10)
            procs = [int(v,10) for v in fields[1:-2]]
            type = fields[-2]
            name = fields[-1]

            mv = max(procs)
            mi = procs.index(mv)
            irqs[irq] = {'cpu':mi, 'type':type, 'name':name, 'count':mv}

    print("Interface: %s" % args.interface)
    print("%4s  %16s  %16s  %7s  %7s" % ('IRQ', 'Name', 'Type', 'Old CPU', 'New CPU')  )
    for i,irq in enumerate(sorted(irqs.keys())):
        oCPU = irqs[irq]['cpu']
        nCPU = args.cpu[i % len(args.cpu)]

        print("%4i  %16s  %16s  %7i  %7i" % (irq, irqs[irq]['name'], irqs[irq]['type'], oCPU, nCPU))

        mask = compute_mask(nCPU)
        write_irq_smp_affinity(irq, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Configure the interrupt request (IRQ) bindings for a particular network interface',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('interface', type=str,
                        help='interface to configure')
    parser.add_argument('cpu', type=int, nargs='+',
                        help='CPU to bind to')
    args = parser.parse_args()
    main(args)
    
