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
        print("  CUDA architectures: %s" % BF_GPU_ARCHS)
        print("  CUDA shared memory: %i B" % BF_GPU_SHAREDMEM)
        print("  CUDA managed memory support: %s" % _yes_no(BF_GPU_MANAGEDMEM))
        print("  CUDA debugging: %s" % _yes_no(BF_CUDA_DEBUG_ENABLED))
        print("  CUDA tracing enabled: %s" % _yes_no(BF_TRACE_ENABLED))
