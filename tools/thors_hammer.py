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
from bifrost.packet_writer import HeaderInfo, DiskWriter

def data_size(string):
    """
    Convert a string like '100GiB' into a size in bytes.
    """
    
    if string.endswith('T') or string.endswith('TiB'):
        value, _ = string.split('T', 1)
        value = int(value) * 1024**4
    elif string.endswith('G') or string.endswith('GiB'):
        value, _ = string.split('G', 1)
        value = int(value) * 1024**3
    elif string.endswith('M') or string.endswith('MiB'):
        value, _ = string.split('M', 1)
        value = int(value) * 1024**2
    elif string.endswith('k') or string.endswith('kiB'):
        value, _ = string.split('k', 1)
        value = int(value) * 1024
    else:
        try:
            value = int(string)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Cannot parse '{string}' as a data size")
    return value

def inverse_data_size(value, include_space=True):
    """
    Invert the data_size() function to convert a size in bytes into a
    data size string.
    """
    
    if value >= 1024**4:
        string = f"{value/1024**4:.1f} TiB"
    elif value >= 1024**3:
        string = f"{value/1024**3:.1f} GiB"
    elif value >= 1024**2:
        string = f"{value/1024**2:.1f} MiB"
    elif value >= 1024:
        string = f"{value/1024:.1f} kiB"
    else:
        string = f"{value:.1f} B"
    if not include_space:
        string = string.replace(' ', '')
    return string

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
    data = np.empty(shape=(args.repeats, args.npacket, 1, args.packet_size), dtype=np.uint8)
    data = BFArray(data, space='system')
    
    # Setup the file flags, enabling O_DIRECT | O_SYNC as needed
    file_flags = os.O_CREAT | os.O_TRUNC | os.O_WRONLY
    if args.direct_io:
        file_flags |= os.O_DIRECT | os.O_SYNC
        
        ## Make sure that the request write width matches what O_DIRECT expects
        if args.packet_size % 512 != 0:
            raise RuntimeError(f"packet_size % 512 != 0")
            
    # Header descriptor (not populated in the case of 'generic'
    desc = HeaderInfo()
    
    # Main loop
    t0 = time.time()
    nbyte = 0
    for i in range(args.file_count):
        t0F = time.time()
        nbyteF = 0
        
        ## Open the i-th file
        fd = os.open(f"{args.filename}.{i+1}", file_flags, mode=0o664)
        fh = DummyFileHandle(fd)
        udt = DiskWriter(f"generic_{args.packet_size}", fh)
        
        ## Write a collection of repeats to the file
        for j in range(args.repeats):
            udt.send(desc, 0, 1, 0, 1, data[j,...])
            nbyteF += args.npacket*args.packet_size
            
        ## Close it out
        del udt
        fh.close()
        
        ## Report the throughput for this file
        t1F = time.time()
        print(f"  #{i+1} in {t1F-t0F:.3f}s -> {inverse_data_size(nbyteF/(t1F-t0F))}/s")
        
        nbyte += nbyteF
        
    # Final throughput report across all file_count files
    t1 = time.time()
    print(f"Finished in {t1-t0:.3f}s -> {inverse_data_size(nbyte/(t1-t0))}/s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='test disk throughput as a function of different Bifrost parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-p', '--packet-size', type=data_size, default=67584,
                        help='number of bytes to write per packet')
    parser.add_argument('-n', '--npacket', type=int, default=1920,
                        help='number of packets to write per Bifrost DiskWriter() call')
    parser.add_argument('-r', '--repeats', type=int, default=1,
                        help='number of times to repeat the npacket x packet-size sized writes')
    parser.add_argument('-c', '--file-count', type=int, default=1,
                        help='number of files containing repeats x npacket x packet-size to write')
    parser.add_argument('-d', '--direct-io', action='store_true',
                        help='enable O_DIRECT | O_SYNC when opening the file(s)')
    parser.add_argument('-f', '--filename', type=str, default='hammer.test',
                        help='base filename to write to')
    args = parser.parse_args()
    main(args)
