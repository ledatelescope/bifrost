#!/usr/bin/env python3

# Copyright (c) 2024, The Bifrost Authors. All rights reserved.
# Copyright (c) 2024, The University of New Mexico. All rights reserved.
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
import time
import numpy as np
import argparse

import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.packet_writer import HeaderInfo, DiskWriter

class DummyFileHandle:
    """
    Wrapper around a Python file descriptor as returned by os.open() to make
￼    it work with Bifrost's DiskWriter.
    """
    
    def __init__(self, fileno):
        self._fileno = fileno

    def fileno(self):
        return self._fileno
        
    def close(self):
        return os.close(self._fileno)

def main(args):
    # Create an empty data array
    data = np.empty(shape=(args.repeats, args.ntime, 1, args.bytes_per_time), dtype=np.uint8)
    data = BFArray(data, space='system')
    
    # Setup the file flags, enabling O_DIRECT | O_SYNC as needed
    file_flags = os.O_CREAT | os.O_TRUNC | os.O_WRONLY
    if args.direct_io:
        file_flags |= os.O_DIRECT | os.O_SYNC
        
        ## Make sure that the request write width matches what O_DIRECT expects
        if args.ntime*args.bytes_per_time % 512 != 0:
            raise RuntimeError(f"ntime x bytes_per_time ({args.ntime*args.bytes_per_time}) % 512 != 0")
            
    # Header descriptor (not populated in the case of 'generic'
    desc = HeaderInfo()
    
    # Main loop
    t0 = time.time()
    nbyte = 0
    for i in range(args.file_count):
        t0F = time.time()
        nbyteF = 0
        
        ## Open the i-th file
        fd = os.open(f"{args.filename}_{i+1}", file_flags, mode=0o664)
        fh = DummyFileHandle(fd)
        udt = DiskWriter(f"generic_{args.bytes_per_time}", fh)
        
        ## Write a collection of repeats to the file
        for j in range(args.repeats):
            udt.send(desc, 0, 1, 0, 1, data[j,...])
            nbyteF += args.ntime*args.bytes_per_time
            
        ## Close it out
        del udt
        fh.close()
        
        ## Report the throughput for this file
        t1F = time.time()
        print(f"  #{i+1} in {t1F-t0F:.3f}s -> {nbyteF/1e9/(t1F-t0F):.1f} GB/s")
        
        nbyte += nbyteF
        
    # Final throughput report across all file_count files
    t1 = time.time()
    print(f"Finished in {t1-t0:.3f}s -> {nbyte/1e9/(t1-t0):.1f} GB/s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='test disk throughput as a function of different Bifrost parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-b', '--bytes-per-time', type=int, default=67584,
                        help='number of bytes to write per time sample')
    parser.add_argument('-n', '--ntime', type=int, default=1920,
                        help='number of time samples to write per Bifrost DiskWriter() call')
    parser.add_argument('-r', '--repeats', type=int, default=1,
                        help='number of times to repeat the ntime x bytes-per-time writes')
    parser.add_argument('-c', '--file-count', type=int, default=1,
                        help='number of files containing repeats x ntime x bytes-per-time to write')
    parser.add_argument('-d', '--direct-io', action='store_true',
                        help='enable O_DIRECT | O_SYNC when opening the file(s)')
    parser.add_argument('-f', '--filename', type=str, default='hammer.test',
                        help='base filename to write to')
    args = parser.parse_args()
    main(args)
