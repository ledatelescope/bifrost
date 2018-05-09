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
import curses
import getopt
import socket
import traceback
import subprocess
try:
    import cStringIO as StringIO
except ImportError:
    import StringIO

os.environ['VMA_TRACELEVEL'] = '0'
from bifrost.proclog import load_by_pid


BIFROST_STATS_BASE_DIR = '/dev/shm/bifrost/'

def usage(exitCode=None):
    print """%s - Display perfomance of different blocks in various bifrost processes

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


def _getLoadAverage():
    """
    Query the /proc/loadavg interface to get the 1, 5, and 10 minutes load 
    averages.  The contents of this file is returned as a dictionary.
    """

    data = {'1min':0.0, '5min':0.0, '10min':0.0, 'procTotal':0, 'procRunning':0, 'lastPID':0}

    with open('/proc/loadavg', 'r') as fh:
        line = fh.read()

        fields = line.split(None, 4)
        procs = fields[3].split('/', 1)

        data['1min'] = float(fields[0])
        data['5min'] = float(fields[1])
        data['10min'] = float(fields[2])

        data['procRunning'] = procs[0]
        data['procTotal'] = procs[1]

        data['lastPID'] = fields[4]
    return data

global _CPU_STATE
_CPU_STATE = {}
def _getProcessorUsage():
    """
    Read in the /proc/stat file to return a dictionary of the load on each \
    CPU.  This dictionary also includes an 'avg' entry that gives the average
    across all CPUs.

    NOTE::  In order for this to work a global variable of _CPU_STATE is
           needed to get the CPU usage change between calls.

    NOTE::  Many of these details could be avoided by using something like the
          Python 'psutil' module.
    """

    data = {'avg': {'user':0.0, 'nice':0.0, 'sys':0.0, 'idle':0.0, 'wait':0.0, 'irq':0.0, 'sirq':0.0, 'steal':0.0, 'total':0.0}}

    with open('/proc/stat', 'r') as fh:
        lines = fh.read()
        fh.close()

        for line in lines.split('\n'):
            if line[:3] == 'cpu':
                fields = line.split(None, 10)
                try:
                    cid = int(fields[0][3:], 10)
                except ValueError:
                    cid = 'avg'
                us = float(fields[1])
                ni = float(fields[2])
                sy = float(fields[3])
                id = float(fields[4])
                wa = float(fields[5])
                hi = float(fields[6])
                si = float(fields[7])
                st = float(fields[8])
                try:
                    us -= _CPU_STATE[cid]['us']
                    ni -= _CPU_STATE[cid]['ni']
                    sy -= _CPU_STATE[cid]['sy']
                    id -= _CPU_STATE[cid]['id']
                    wa -= _CPU_STATE[cid]['wa']
                    hi -= _CPU_STATE[cid]['hi']
                    si -= _CPU_STATE[cid]['si']
                    st -= _CPU_STATE[cid]['st']
                except KeyError:
                    _CPU_STATE[cid] = {'us':us, 'ni':ni, 'sy':sy, 'id':id, 'wa':wa, 'hi':hi, 'si':si, 'st':st}

                t = us+ni+sy+id+wa+hi+si+st

                data[cid] = {'user':us/t, 'nice':ni/t, 'sys':sy/t, 'idle':id/t, 
                           'wait':wa/t, 'irq':hi/t, 'sirq':si/t, 'steal':st/t, 
                           'total':(us+ni+sy)/t}
            else:
                break
    return data


def _getMemoryAndSwapUsage():
    """
    Read in the /proc/meminfo and return a dictionary of the memory and swap 
    usage for all processes.

    NOTE::  Many of these details could be avoided by using something like the
          Python 'psutil' module.
    """

    data = {'memTotal':0, 'memUsed':0, 'memFree':0, 
           'swapTotal':0, 'swapUsed':0, 'swapFree':0, 
           'buffers':0, 'cached':0}

    with open('/proc/meminfo', 'r') as fh:
        lines = fh.read()
        fh.close()

        for line in lines.split('\n'):
            fields = line.split(None, 2)
            if fields[0] == 'MemTotal:':
                data['memTotal'] = int(fields[1], 10)
            elif fields[0] == 'MemFree:':
                data['memFree'] = int(fields[1], 10)
            elif fields[0] == 'Buffers:':
                data['buffers'] = int(fields[1], 10)
            elif fields[0] == 'Cached:':
                data['cached'] = int(fields[1], 10)
            elif fields[0] == 'SwapTotal:':
                data['swapTotal'] = int(fields[1], 10)
            elif fields[0] == 'SwapFree:':
                data['swapFree'] = int(fields[1], 10)
                break
        data['memUsed'] = data['memTotal'] - data['memFree']
        data['swapUsed'] = data['swapTotal'] - data['swapFree']
    return data


def _getGpuMemoryUsage():
    """
    Grab nvidia-smi output and return a dictionary of the memory usage.
    """
    
    data = {'devCount':0, 'memTotal':0, 'memUsed':0, 'memFree':0, 'pwrDraw':0.0, 'pwrLimit':0.0, 'load':0.0}
    
    q_flag   = '--query-gpu=memory.used,memory.total,memory.free,power.draw,power.limit,utilization.gpu'
    fmt_flag = '--format=csv,noheader,nounits'
    try:
        p = subprocess.Popen(['nvidia-smi', q_flag, fmt_flag], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate()
    except (OSError, ValueError) as e:
        pass
    else:
        # Parse the ouptut and turn everything into something useful, if possible
        lines = output.split('\n')[:-1]
        for line in lines:
            used, total, free, draw, limit, load = line.split(',')
            data['devCount'] += 1
            data['memTotal'] += int(total, 10)*1024
            data['memUsed']  += int(used, 10)*1024
            data['memFree']  += int(free, 10)*1024
            try:
                data['pwrDraw'] += float(draw)
                data['pwrLimit'] += float(limit)
            except ValueError:
                pass
            try:
                data['load'] += float(load)
            except ValueError:
                pass
        # Convert the load to an average
        data['load'] /= data['devCount']
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


def _addLine(screen, y, x, string, *args):
    """
    Helper function for curses to add a line, clear the line to the end of 
    the screen, and update the line number counter.
    """

    screen.addstr(y, x, string, *args)
    screen.clrtoeol()
    return y + 1


_REDRAW_INTERVAL_SEC = 0.2


def main(args):
    config = parseOptions(args)

    hostname = socket.gethostname()

    scr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    scr.keypad(1)
    scr.nodelay(1)
    size = scr.getmaxyx()

    std = curses.A_NORMAL
    rev = curses.A_REVERSE

    poll_interval = 1.0
    tLastPoll = 0.0
    sort_key = 'process'
    sort_rev = True
    display_gpu = False

    try:
        while True:
            t = time.time()

            ## Interact with the user
            c = scr.getch()
            curses.flushinp()
            if c == ord('q'):
                break
            elif c == ord('i'):
                new_key = 'pid'
            elif c == ord('b'):
                new_key = 'name'
            elif c == ord('c'):
                new_key = 'core'
            elif c == ord('t'):
                new_key = 'total'
            elif c == ord('a'):
                new_key = 'acquire'
            elif c == ord('p'):
                new_key = 'process'
            elif c == ord('r'):
                new_key = 'reserve'

            try:
                if sort_key == new_key:
                    sort_rev = not sort_rev
                else:
                    sort_key = new_key
                    sort_rev = True
                del new_key
            except NameError:
                pass

            ## Do we need to poll the system again?
            if t-tLastPoll > poll_interval:
                ## Load in the various bits form /proc that we need
                load = _getLoadAverage()
                cpu  = _getProcessorUsage()
                mem  = _getMemoryAndSwapUsage()
                gpu  = _getGpuMemoryUsage()
                
                ## Determine if we have GPU data to display
                if gpu['devCount'] > 0:
                    display_gpu = True
                    
                ## Find all running processes
                pidDirs = glob.glob(os.path.join(BIFROST_STATS_BASE_DIR, '*'))
                pidDirs.sort()

                ## Load the data
                blockList = {}
                for pidDir in pidDirs:
                    pid = int(os.path.basename(pidDir), 10)
                    contents = load_by_pid(pid)

                    cmd = _getCommandLine(pid)
                    if cmd == '':
                        continue

                    for block in contents.keys():
                        try:
                            log = contents[block]['bind']
                            cr = log['core0']
                        except KeyError:
                            continue

                        try:
                            log = contents[block]['perf']
                            ac = max([0.0, log['acquire_time']])
                            pr = max([0.0, log['process_time']])
                            re = max([0.0, log['reserve_time']])
                        except KeyError:
                            ac, pr, re = 0.0, 0.0, 0.0

                        blockList['%i-%s' % (pid, block)] = {'pid': pid, 'name':block, 'cmd': cmd, 'core': cr, 'acquire': ac, 'process': pr, 'reserve': re, 'total':ac+pr+re}

                ## Sort
                order = sorted(blockList, key=lambda x: blockList[x][sort_key], reverse=sort_rev)

                ## Mark
                tLastPoll = time.time()

            ## Display
            k = 0
            ### General - load average
            output = '%s - %s - load average: %s, %s, %s\n' % (os.path.basename(__file__), hostname, load['1min'], load['5min'], load['10min'])
            k = _addLine(scr, k, 0, output, std)
            ### General - process counts
            output = 'Processes: %s total, %s running\n' % (load['procTotal'], load['procRunning'])
            k = _addLine(scr, k, 0, output, std)
            ### General - average processor usage
            c = cpu['avg']
            output = 'CPU(s):%5.1f%%us,%5.1f%%sy,%5.1f%%ni,%5.1f%%id,%5.1f%%wa,%5.1f%%hi,%5.1f%%si,%5.1f%%st\n' % (100.0*c['user'], 100.0*c['sys'], 100.0*c['nice'], 100.0*c['idle'], 100.0*c['wait'], 100.0*c['irq'], 100.0*c['sirq'], 100.0*c['steal'])
            k = _addLine(scr, k, 0, output, std)
            ### General - memory
            output = 'Mem:    %9ik total, %9ik used, %9ik free, %9ik buffers\n' % (mem['memTotal'], mem['memUsed'], mem['memFree'], mem['buffers'])
            k = _addLine(scr, k, 0, output, std)
            ### General - swap
            output = 'Swap:   %9ik total, %9ik used, %9ik free, %9ik cached\n' % (mem['swapTotal'], mem['swapUsed'], mem['swapFree'], mem['cached'])
            k = _addLine(scr, k, 0, output, std)
            ### General - GPU, if avaliable
            if display_gpu:
                if gpu['pwrLimit'] != 0.0:
                    if gpu['load'] != 0.0:
                        output = 'GPU(s): %9ik total, %9ik used, %9ik free, %5.1f%%us, %.0f/%.0fW\n' % (gpu['memTotal'], gpu['memUsed'], gpu['memFree'], gpu['load'], gpu['pwrDraw'], gpu['pwrLimit'])
                    else:
                        output = 'GPU(s): %9ik total, %9ik used, %9ik free, %.0f/%.0fW\n' % (gpu['memTotal'], gpu['memUsed'], gpu['memFree'], gpu['pwrDraw'], gpu['pwrLimit'])
                else:
                    output = 'GPU(s): %9ik total, %9ik used, %9ik free, %i device(s)\n' % (gpu['memTotal'], gpu['memUsed'], gpu['memFree'], gpu['devCount'])
                k = _addLine(scr, k, 0, output, std)
            ### Header
            k = _addLine(scr, k, 0, ' ', std)
            output = '%6s  %15s  %4s  %5s  %7s  %7s  %7s  %7s  Cmd' % ('PID', 'Block', 'Core', '%CPU', 'Total', 'Acquire', 'Process', 'Reserve')
            csize = size[1]-len(output)
            output += ' '*csize
            output += '\n'
            k = _addLine(scr, k, 0, output, rev)
            ### Data
            for o in order:
                d = blockList[o]
                try:
                    c = 100.0*cpu[d['core']]['total']
                    c = '%5.1f' % c
                except KeyError:
                    c = '%5s' % ' '
                output = '%6i  %15s  %4i  %5s  %7.3f  %7.3f  %7.3f  %7.3f  %s' % (d['pid'], d['name'][:15], d['core'], c, d['total'], d['acquire'], d['process'], d['reserve'], d['cmd'][:csize+3])
                k = _addLine(scr, k, 0, output, std)
                if k >= size[0] - 1:
                    break
            ### Clear to the bottom
            scr.clrtobot()
            ### Refresh
            scr.refresh()

            ## Sleep
            time.sleep(_REDRAW_INTERVAL_SEC)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fileObject = StringIO.StringIO()
        traceback.print_tb(exc_traceback, file=fileObject)
        tbString = fileObject.getvalue()
        fileObject.close()

    # Save the window contents
    contents = ''
    y,x = scr.getmaxyx()
    for i in xrange(y-1):
        for j in xrange(x):
            d = scr.inch(i,j)
            c = d&0xFF
            a = (d>>8)&0xFF
            contents += chr(c)

    # Tear down curses
    scr.keypad(0)
    curses.echo()
    curses.nocbreak()
    curses.endwin()
    
    # Final reporting
    try:
        ## Error
        print "%s: failed with %s at line %i" % (os.path.basename(__file__), str(error), traceback.tb_lineno(exc_traceback))
        for line in tbString.split('\n'):
            print line
    except NameError:
        ## Last window contents sans attributes
        print contents


if __name__ == "__main__":
    main(sys.argv[1:])
    
