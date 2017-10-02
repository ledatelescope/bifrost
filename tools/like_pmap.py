#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import re
import sys
import getopt
import subprocess

os.environ['VMA_TRACELEVEL'] = '0'
from bifrost.proclog import load_by_pid


def usage(exitCode=None):
    print """%s - Get a detailed look at memory usage in a bifrost pipeline

Usage: %s [OPTIONS] pid

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


def _getBestSize(value):
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
    config = parseOptions(args)
    pidToUse = int(config['args'][0], 10)
    
    # Find out the kernel page size, both regular and huge
    ## Regular
    pageSize = subprocess.check_output(['getconf', 'PAGESIZE'])
    pageSize = int(pageSize, 10)
    ## Huge - assumed that the value is in kB
    hugeSize = subprocess.check_output(['grep', 'Hugepagesize', '/proc/meminfo'])
    hugeSize = int(hugeSize.split()[1], 10) * 1024
    
    # Load in the bifrost ring information for this process
    contents = load_by_pid(pidToUse, include_rings=True)
    rings = {}
    for block in contents.keys():
        if block == 'rings':
            for ring in contents[block].keys():
                rings[ring] = {}
                for key in contents[block][ring]:
                    rings[ring][key] = contents[block][ring][key]
                continue
    if not rings:
        raise RuntimeError("Cannot find bifrost ring info for PID: %i" % pidToUse)
        
    # Load in the NUMA map page for this process
    try:
        fh = open('/proc/%i/numa_maps' % pidToUse, 'r')
        numaInfo = fh.read()
        fh.close()
    except IOError:
        raise RuntimeError("Cannot find NUMA memory info for PID: %i" % pidToUse)
        
    # Parse out the anonymous entries in this file
    _numaRE = re.compile('(?P<addr>[0-9a-f]+).*[(anon)|(mapped)]=(?P<size>\d+).*(swapcache=(?P<swap>\d+))?.*N(?P<binding>\d+)=(?P<size2>\d+)')
    areas = {}
    files = {}
    for line in numaInfo.split('\n'):
        ## Skp over blank lines, files, and anything that is not anonymous
        if len(line) < 3:
            continue
        elif line.find('file=') != -1:
            ## Run  regex over the line to get the address, size, and binding information
            mtch = _numaRE.search(line)
            if mtch is not None:
                ### Basic info
                heap  = True if line.find('heap') != -1 else False
                stack = True if line.find('stack') != -1 else False
                huge  = True if line.find('huge') != -1 else False
                share = True if line.find('mapmax=') != -1 else False
                
                ### Detailed info
                addr  = mtch.group('addr')
                size  = int(mtch.group('size'), 10)
                size *= hugeSize if huge else pageSize
                try:
                    ssize = int(mtch.group('swap'), 10)
                    swap = True
                except TypeError:
                    ssize = 0
                    swap = False
                ssize *=  hugeSize if huge else pageSize
                node = int(mtch.group('binding'), 10)
                
                ### Save
                files[addr] = {'size':size, 'node':node, 'huge':huge, 'heap':heap, 'stack':stack, 'shared':share, 'swapped':swap, 'swapsize':ssize}
                
        elif line.find('anon=') != -1:
            ## Run  regex over the line to get the address, size, and binding information
            mtch = _numaRE.search(line)
            if mtch is not None:
                ### Basic info
                heap  = True if line.find('heap') != -1 else False
                stack = True if line.find('stack') != -1 else False
                huge  = True if line.find('huge') != -1 else False
                share = True if line.find('mapmax=') != -1 else False
                
                ### Detailed info
                addr  = mtch.group('addr')
                size  = int(mtch.group('size'), 10)
                size *= hugeSize if huge else pageSize
                try:
                    ssize = int(mtch.group('swap'), 10)
                    swap = True
                except TypeError:
                    ssize = 0
                    swap = False
                ssize *=  hugeSize if huge else pageSize
                node = int(mtch.group('binding'), 10)
                
                ### Save
                areas[addr] = {'size':size, 'node':node, 'huge':huge, 'heap':heap, 'stack':stack, 'shared':share, 'swapped':swap, 'swapsize':ssize}
            
    # Try to match the rings to the memory areas
    matched = []
    for ring in rings:
        stride = rings[ring]['stride']
        
        best   = None
        metric = 1e13
        for addr in areas:
            diff = abs(areas[addr]['size'] - stride)
            if diff < metric:
                best = addr
                metric = diff
        rings[ring]['addr'] = best
        matched.append( best )
        
    # Take a look at how the areas are bound
    nodeCountsAreas = {}
    nodeSizesAreas = {}
    for addr in areas:
        node = areas[addr]['node']
        size = areas[addr]['size']
        try:
            nodeCountsAreas[node] += 1
            nodeSizesAreas[node] += size
        except KeyError:
            nodeCountsAreas[node] = 1
            nodeSizesAreas[node] = size
    nodeCountsFiles = {}
    nodeSizesFiles = {}
    for addr in files:
        node = files[addr]['node']
        size = files[addr]['size']
        try:
            nodeCountsFiles[node] += 1
            nodeSizesFiles[node] += size
        except KeyError:
            nodeCountsFiles[node] = 1
            nodeSizesFiles[node] = size
            
    # Final report
    print "Rings: %i" % len(rings)
    print "File Backed Memory Areas:"
    print "  Total: %i" % len(files)
    print "  Heap: %i" % len([addr for addr in files if files[addr]['heap']])
    print "  Stack: %i" % len([addr for addr in files if files[addr]['stack']])
    print "  Shared: %i" % len([addr for addr in files if files[addr]['shared']])
    print "  Swapped: %i" % len([addr for addr in files if files[addr]['swapped']])
    for node in sorted(nodeCountsFiles.keys()):
        print "  NUMA Node %i:" % node
        print "    Count: %i" % nodeCountsFiles[node]
        print "    Size: %.3f %s" % _getBestSize(nodeSizesFiles[node])
    print "Anonymous Memory Areas:"
    print "  Total: %i" % len(areas)
    print "  Heap: %i" % len([addr for addr in areas if areas[addr]['heap']])
    print "  Stack: %i" % len([addr for addr in areas if areas[addr]['stack']])
    print "  Shared: %i" % len([addr for addr in areas if areas[addr]['shared']])
    print "  Swapped: %i" % len([addr for addr in areas if areas[addr]['swapped']])
    for node in sorted(nodeCountsAreas.keys()):
        print "  NUMA Node %i:" % node
        print "    Count: %i" % nodeCountsAreas[node]
        print "    Size: %.3f %s" % _getBestSize(nodeSizesAreas[node])
    print " "
    
    print "Ring Mappings:"
    for ring in sorted(rings):
        print "  %s" % ring
        try:
            area = areas[rings[ring]['addr']]
        except KeyError:
            print "    Unknown"
            continue
        sv, su = _getBestSize(area['size'])
        diff = abs(area['size'] - rings[ring]['stride'])
        status = ''
        if diff > 0.5*hugeSize:
            status = '???'
        dv, du = _getBestSize(diff)
        sf = float(area['swapsize'])/float(area['size'])
        
        print "    Size: %.3f %s" % _getBestSize(rings[ring]['stride'])
        print "    Area: %s %s" % (rings[ring]['addr'], status)
        print "      Size: %.3f %s%s" % (sv, su, ' (within %.3f %s)' % (dv, du) if diff != 0 else '')
        print "      Node: %i" % area['node']
        print "      Attributes:"
        print "        Huge? %s" % area['huge']
        print "        Heap? %s" % area['heap']
        print "        Stack? %s" % area['stack']
        print "        Shared? %s" % area['shared']
        print "      Swap Status:"
        print "        Swapped? %s" % area['swapped']
        if area['swapped']:
            print "        Swap Fraction: %.1f%%" % (100.0*sf,)
    print " "
    
    print "Other Non-Ring Areas:"
    print "  Size: %.3f %s" % _getBestSize(sum([areas[area]['size'] for area in areas if area not in matched]))
    print " "
    
    print "File Backed Areas:"
    print "  Size: %.3f %s" % _getBestSize(sum([files[area]['size'] for area in files]))


if __name__ == "__main__":
    main(sys.argv[1:])
    