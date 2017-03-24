#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import curses
import getopt

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


def _getLoadAverage():
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


def _getCommandLine(pid):
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
	screen.addstr(y, x, string, *args)
	screen.clrtoeol()
	return y + 1


def main(args):
	config = parseOptions(args)
	
	scr = curses.initscr()
	curses.noecho()
	curses.cbreak()
	scr.keypad(1)
	scr.nodelay(1)
	size = scr.getmaxyx()
	
	std = curses.A_NORMAL
	rev = curses.A_REVERSE
	
	try:
		while True:
			t = time.time()
			
			## Interact with the user
			c = scr.getch()
			curses.flushinp()
			if c == ord('q'):
				break
				
			## Load in the various bits form /proc that we need
			load = _getLoadAverage()
			cpu  = _getProcessorUsage()
			mem  = _getMemoryAndSwapUsage()
			
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
						
					blockList['%i-%s' % (pid, block)] = {'pid': pid, 'name':block, 'cmd': cmd, 'core': cr, 'acquire': ac, 'process': pr, 'reserve': re}
					
			## Sort
			order = sorted(blockList, key=lambda x: blockList[x]['process'], reverse=True)
			
			## Display
			k = 0
			### General - load average
			output = '%s - load average: %s, %s, %s\n' % (os.path.basename(__file__), load['1min'], load['5min'], load['10min'])
			k = _addLine(scr, k, 0, output, std)
			### General - process counts
			output = 'Processes: %s total, %s running\n' % (load['procTotal'], load['procRunning'])
			k = _addLine(scr, k, 0, output, std)
			### General - average processor usage
			c = cpu['avg']
			output = 'CPU(s):%5.1f%%us,%5.1f%%sy,%5.1f%%ni,%5.1f%%id,%5.1f%%wa,%5.1f%%hi,%5.1f%%si,%5.1f%%st\n' % (100.0*c['user'], 100.0*c['sys'], 100.0*c['nice'], 100.0*c['idle'], 100.0*c['wait'], 100.0*c['irq'], 100.0*c['sirq'], 100.0*c['steal'])
			k = _addLine(scr, k, 0, output, std)
			### General - memory
			output = 'Mem:  %9ik total, %9ik used, %9ik free, %9ik buffers\n' % (mem['memTotal'], mem['memUsed'], mem['memFree'], mem['buffers'])
			k = _addLine(scr, k, 0, output, std)
			### General - swap
			output = 'Swap: %9ik total, %9ik used, %9ik free, %9ik cached\n' % (mem['swapTotal'], mem['swapUsed'], mem['swapFree'], mem['cached'])
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
				output = '%6i  %15s  %4i  %5s  %7.3f  %7.3f  %7.3f  %7.3f  %s' % (d['pid'], d['name'], d['core'], c, d['acquire']+d['process']+d['reserve'], d['acquire'], d['process'], d['reserve'], d['cmd'][:csize+3])
				k = _addLine(scr, k, 0, output, std)
				if k > size[0]:
					break
			### Clear to the bottom
			scr.clrtobot()
			### Refresh
			scr.refresh()
			
			## Sleep
			time.sleep(3.0)
			
	except KeyboardInterrupt:
		pass
		
	curses.nocbreak()
	scr.keypad(0)
	curses.echo()
	curses.endwin()


if __name__ == "__main__":
	main(sys.argv[1:])
	
