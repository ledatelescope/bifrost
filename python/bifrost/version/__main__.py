
# Copyright (c) 2021-2022, The Bifrost Authors. All rights reserved.
# Copyright (c) 2021-2022, The University of New Mexico. All rights reserved.
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

from __future__ import print_function

import argparse

from bifrost import __version__, __copyright__, __license__
from bifrost.libbifrost_generated import *

def _yes_no(value):
    return "yes" if value else "no"

parser = argparse.ArgumentParser(description='Bifrost version/configuration information')
parser.add_argument('--config', action='store_true',
                    help='also display configuration information')
args = parser.parse_args()

print("\n".join(["bifrost " + __version__, __copyright__, "License: " + __license__]))
if args.config:
    print("\nConfiguration:")
    print(" Memory alignment: %i B" % BF_ALIGNMENT)
    print(" OpenMP support: %s" % _yes_no(BF_OPENMP_ENABLED))
    print(" NUMA support %s" % _yes_no(BF_NUMA_ENABLED))
    print(" Hardware locality support: %s" % _yes_no(BF_HWLOC_ENABLED))
    print(" Mellanox messaging accelerator (VMA) support: %s" % _yes_no(BF_VMA_ENABLED))
    print(" Logging directory: %s" % BF_PROCLOG_DIR)
    print(" Debugging: %s" % _yes_no(BF_DEBUG_ENABLED))
    print(" CUDA support: %s" % _yes_no(BF_CUDA_ENABLED))
    if BF_CUDA_ENABLED:
        print("  CUDA version: %.1f" % BF_CUDA_VERSION)
        print("  CUDA architectures: %s" % BF_GPU_ARCHS)
        print("  CUDA shared memory: %i B" % BF_GPU_SHAREDMEM)
        print("  CUDA managed memory support: %s" % _yes_no(BF_GPU_MANAGEDMEM))
        print("  CUDA map disk cache: %s" % _yes_no(BF_MAP_KERNEL_DISK_CACHE))
        print("  CUDA debugging: %s" % _yes_no(BF_CUDA_DEBUG_ENABLED))
        print("  CUDA tracing enabled: %s" % _yes_no(BF_TRACE_ENABLED))
