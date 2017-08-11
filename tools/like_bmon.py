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
from datetime import datetime
try:
    import cStringIO as StringIO
except ImportError:
    import StringIO

os.environ['VMA_TRACELEVEL'] = '0'
from bifrost.proclog import load_by_pid


BIFROST_STATS_BASE_DIR = '/dev/shm/bifrost/'

def usage(exitCode=None):
    print """%s - Monitor the packets capture/transmit status of a 
bifrost pipeline.

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


def _getTransmitReceive():
    """
    Read in the /dev/bifrost ProcLog data and return block-level information 
    about udp* blocks.
    """

    ## Find all running processes
    pidDirs = glob.glob(os.path.join(BIFROST_STATS_BASE_DIR, '*'))
    pidDirs.sort()

    ## Load the data
    blockList = {}
    for pidDir in pidDirs:
        pid = int(os.path.basename(pidDir), 10)
        contents = load_by_pid(pid)

        for block in contents.keys():
            if block[:3] != 'udp':
                continue

            t = time.time()
            try:
                log     = contents[block]['stats']
                good    = log['ngood_bytes']
                missing = log['nmissing_bytes']
                invalid = log['ninvalid_bytes']
                late    = log['nlate_bytes']
                nvalid  = log['nvalid']
            except KeyError:
                good, missing, invalid, late, nvalid = 0, 0, 0, 0, 0

            blockList['%i-%s' % (pid, block)] = {'pid': pid, 'name':block, 
                                          'time':t, 
                                          'good': good, 'missing': missing, 
                                          'invalid': invalid, 'late': late, 
                                          'nvalid': nvalid}
    return blockList


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


def _getStatistics(blockList, prevList):
    """
    Given a list of running blocks and a previous version of that, compute 
    basic statistics for the UDP blocks.
    """

    # Loop over the blocks to find udp_capture and udp_transmit blocks
    output = {'updated': datetime.now()}
    for block in blockList:
        if block.find('udp_capture') != -1:
            ## udp_capture is RX
            good = True
            type = 'rx'
            curr = blockList[block]
            try:
                prev = prevList[block]
            except KeyError:
                prev = curr

        elif block.find('udp_transmit') != -1:
            ## udp_transmit is TX
            good = True
            type = 'tx'
            curr = blockList[block]
            try:
                prev = prevList[block]
            except KeyError:
                prev = curr

        else:
            ## Other is not relevant
            good = False

        ## Skip over irrelevant blocks
        if not good:
            continue

        ## PID
        pid = curr['pid']
        ## Computed statistics - rates
        try:
            drate = (curr['good'  ] - prev['good'  ]) / (curr['time'] - prev['time'])
            prate = (curr['nvalid'] - prev['nvalid']) / (curr['time'] - prev['time'])
        except ZeroDivisionError:
            drate = 0.0
            prate = 0.0
        ## Computed statistics - packet loss - global
        try:
            gloss = 100.0*curr['missing']/(curr['good'] + curr['missing'])
        except ZeroDivisionError:
            gloss = 0.0
        ## Computed statistics - packet loss - current
        try:
            closs = 100.0*(curr['missing']-prev['missing'])/(curr['missing']-prev['missing']+curr['good']-prev['good'])
        except ZeroDivisionError:
            closs = 0.0

        ## Save
        ### Setup
        try:
            output[pid][type]
        except KeyError:
            output[pid] = {}
            output[pid]['rx' ] = {'good':0, 'missing':0, 'invalid':0, 'late':0, 'drate':0.0, 'prate':0.0, 'gloss':0.0, 'closs':0.0}
            output[pid]['tx' ] = {'good':0, 'missing':0, 'invalid':0, 'late':0, 'drate':0.0, 'prate':0.0, 'gloss':0.0, 'closs':0.0}
            output[pid]['cmd'] = _getCommandLine(pid)
        ### Actual data
        output[pid][type]['good'   ] = curr['good'   ]
        output[pid][type]['missing'] = curr['missing']
        output[pid][type]['invalid'] = curr['invalid']
        output[pid][type]['late'   ] = curr['late'   ]
        output[pid][type]['drate'  ] = max([0.0, drate])
        output[pid][type]['prate'  ] = max([0.0, prate])
        output[pid][type]['gloss'  ] = max([0.0, min([gloss, 100.0])])
        output[pid][type]['closs'  ] = max([0.0, min([closs, 100.0])])

    # Done
    return output


def _setUnits(value):
    """
    Convert a value in bytes so a human-readable format with units.
    """

    if value > 1024.0**3:
        value = value / 1024.0**3
        unit = 'GB'
    elif value > 1024.0**2:
        value = value / 1024.0**2
        unit = 'MB'
    elif value > 1024.0**1:
        value = value / 1024.0*1
        unit = 'kB'
    else:
        unit = ' B'
    return value, unit


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

    blockList = _getTransmitReceive()
    order = sorted([blockList[key]['pid'] for key in blockList])
    order = set(order)
    nPID = len(order)

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

    try:
        sel = 0

        while True:
            t = time.time()

            ## Interact with the user
            c = scr.getch()
            curses.flushinp()
            if c == ord('q'):
                break
            elif c == curses.KEY_UP:
                sel -= 1
            elif c == curses.KEY_DOWN:
                sel += 1

            ## Find the current selected process and see if it has changed
            newSel = min([nPID-1, max([0, sel])])
            if newSel != sel:
                tLastPoll = 0.0
                sel = newSel

            ## Do we need to poll the system again?
            if t-tLastPoll > poll_interval:
                ## Save what we had before
                prevList = blockList

                ## Find all running processes
                pidDirs = glob.glob(os.path.join(BIFROST_STATS_BASE_DIR, '*'))
                pidDirs.sort()

                ## Load the data
                blockList = _getTransmitReceive()

                ## Sort
                order = sorted([blockList[key]['pid'] for key in blockList])
                order = list(set(order))
                nPID = len(order)

                ## Stats
                stats = _getStatistics(blockList, prevList)

                ## Mark
                tLastPoll = time.time()

                ## Clear
                act = None

            ## For sel to be valid - this takes care of any changes between when 
            ## we get what to select and when we polled the bifrost logs
            sel = min([nPID-1, sel])

            ## Display
            k = 0
            ### General - selected
            try:
                output = ' PID: %i on %s' % (order[sel], hostname)
            except IndexError:
                output = ' PID: n/a on %s' % (hostname,)
            output += ' '*(size[1]-len(output)-len(os.path.basename(__file__))-1)
            output += os.path.basename(__file__)+' '
            output += '\n'
            k = _addLine(scr, k, 0, output, std)
            ### General - header
            k = _addLine(scr, k, 0, ' ', std)
            output = '%6s        %9s        %6s        %9s        %6s' % ('PID', 'RX Rate', 'RX #/s', 'TX Rate', 'TX #/s')
            output += ' '*(size[1]-len(output))
            output += '\n'
            k = _addLine(scr, k, 0, output, rev)
            ### Data
            for o in order:
                curr = stats[o]
                if o == order[sel]:
                    act = curr

                drateR, prateR = curr['rx']['drate'], curr['rx']['prate']
                drateR, drateuR = _setUnits(drateR)

                drateT, prateT = curr['tx']['drate'], curr['tx']['prate']
                drateT, drateuT = _setUnits(drateT)


                output = '%6i        %7.2f%2s        %6i        %7.2f%2s        %6i\n' % (o, drateR, drateuR, prateR, drateT, drateuT, prateT)
                try:
                    if o == order[sel]:
                        sty = std|curses.A_BOLD
                    else:
                        sty = std
                except IndexError:
                    sty = std
                k = _addLine(scr, k, 0, output, sty)

                if k > size[0]-9:
                    break
            while k < size[0]-9:
                output = ' '
                k = _addLine(scr, k, 0, output, std)

            ### Details of selected
            output = 'Details - %8s     %19s           %19s' % (stats['updated'].strftime("%H:%M:%S"), 'RX', 'TX')
            output += ' '*(size[1]-len(output))
            output += '\n'
            k = _addLine(scr, k, 0, output, rev)
            if act is not None:
                output = 'Good:                  %18iB           %18iB\n' % (act['rx']['good'   ], act['tx']['good'   ])
                k = _addLine(scr, k, 0, output, std)
                output = 'Missing:               %18iB           %18iB\n' % (act['rx']['missing'], act['tx']['missing'])
                k = _addLine(scr, k, 0, output, std)
                output = 'Invalid:               %18iB           %18iB\n' % (act['rx']['invalid'], act['tx']['invalid'])
                k = _addLine(scr, k, 0, output, std)
                output = 'Late:                  %18iB           %18iB\n' % (act['rx']['late'   ], act['tx']['late'   ])
                k = _addLine(scr, k, 0, output, std)
                output = 'Global Missing:        %18.2f%%           %18.2f%%\n' % (act['rx']['gloss'  ], act['tx']['gloss'  ])
                k = _addLine(scr, k, 0, output, std)
                output = 'Current Missing:       %18.2f%%           %18.2f%%\n' % (act['rx']['closs'  ], act['tx']['closs'  ])
                k = _addLine(scr, k, 0, output, std)
                output = 'Command:               %s' % act['cmd']
                k = _addLine(scr, k, 0, output[:size[1]], std)

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

    scr.keypad(0)
    curses.echo()
    curses.nocbreak()
    curses.endwin()

    try:
        print "%s: failed with %s at line %i" % (os.path.basename(__file__), str(error), traceback.tb_lineno(exc_traceback))
        for line in tbString.split('\n'):
            print line
    except NameError:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
