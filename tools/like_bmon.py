#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import curses
import getopt
import socket

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
		unit = 'B '
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
	
	poll_interval = 3.0
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
				
				## Mark
				tLastPoll = time.time()
				
			## Display
			k = 0
			### General - selected
			output = ' PID: %i on %s' % (order[sel], hostname)
			output += ' '*(size[1]-len(output)-len(os.path.basename(__file__))-1)
			output += os.path.basename(__file__)+' '
			output += '\n'
			k = _addLine(scr, k, 0, output, std)
			### General - header
			k = _addLine(scr, k, 0, ' ', std)
			output = '%6s        %9s        %6s        %9s        %6s' % ('PID', 'RX Rate', 'RX #', 'TX Rate', 'TX #')
			output += ' '*(size[1]-len(output))
			output += '\n'
			k = _addLine(scr, k, 0, output, rev)
			### Data
			for o in order:
				try:
					dt = blockList['%i-udp_transmit' % o]
				except KeyError:
					dt = {}
				try:
					pdt = prevList['%i-udp_transmit' % o]
				except KeyError:
					pdt = {}
				try:
					if o == order[sel]:
						st = dt
						pst = pdt
					try:
						tr = (dt['good'] - pdt['good']) / (dt['time'] - pdt['time'])
					except ZeroDivisionError:
						tr = 0.0
					tp = dt['nvalid'] - pdt['nvalid']
				except KeyError:
					tr = 0.0
					tp = 0
				tr, tu = _setUnits(tr)
				
				
				try:
					dr = blockList['%i-udp_capture' % o]
				except KeyError:
					dr = {}
				try:
					pdr = prevList['%i-udp_capture' % o]
				except KeyError:
					pdr = {}
				try:
					if o == order[sel]:
						sr = dr
						psr = pdr
					try:
						rr = (dr['good'] - pdr['good']) / (dr['time'] - pdr['time'])
					except ZeroDivisionError:
						rr = 0.0
					rp = dr['nvalid'] - pdr['nvalid']
				except KeyError:
					rr = 0.0
					rp = 0
				rr, ru = _setUnits(rr)
				
				
				output = '%6i        %7.2f%2s        %6i        %7.2f%2s        %6i\n' % (o, rr, ru, rp, tr, tu, tp)
				if o == order[sel]:
					k = _addLine(scr, k, 0, output, std|curses.A_BOLD)
				else:
					k = _addLine(scr, k, 0, output, std)
				if k > size[0]-9:
					break
			while k < size[0]-9:
				output = ' '
				k = _addLine(scr, k, 0, output, std)
			### Details of selected
			try:
				gt = st['good']
				it = st['invalid']
				mt = st['missing']
				lt = st['late']
				try:
					ft = 100.0*mt/(mt+gt)
				except ZeroDivisionError:
					ft = 0.0
				try:
					ct = 100.0*(st['missing']-pst['missing'])/(st['missing']-pst['missing']+st['good']-pst['good'])
				except ZeroDivisionError:
					ct = 0.0
			except KeyError:
				gt = 0
				mt = 0
				it = 0
				lt = 0
				ft = 0.0
				ct = 0.0
			try:
				gr = sr['good']
				mr = sr['missing']
				ir = sr['invalid']
				lr = sr['late']
				try:
					fr = 100.0*mr/(mr+gr)
				except ZeroDivisionError:
					ft = 0.0
				try:
					cr = 100.0*(sr['missing']-psr['missing'])/(sr['missing']-psr['missing']+sr['good']-psr['good'])
				except ZeroDivisionError:
					cr = 0.0
			except KeyError:
				gr = 0
				mr = 0
				ir = 0
				lr = 0
				fr = 0.0
				cr = 0.0
			output = 'Details            %19s           %19s' % ('RX', 'TX')
			output += ' '*(size[1]-len(output))
			output += '\n'
			k = _addLine(scr, k, 0, output, rev)
			output = 'Good:              %18iB          %18iB\n' % (gr, gt)
			k = _addLine(scr, k, 0, output, std)
			output = 'Missing:           %18iB          %18iB\n' % (mr, mt)
			k = _addLine(scr, k, 0, output, std)
			output = 'Invalid:           %18iB          %18iB\n' % (ir, it)
			k = _addLine(scr, k, 0, output, std)
			output = 'Late:              %18iB          %18iB\n' % (lr, lt)
			k = _addLine(scr, k, 0, output, std)
			output = 'Global Missing:    %18.2f%%          %18.2f%%\n' % (fr, ft)
			k = _addLine(scr, k, 0, output, std)
			output = 'Current Missing:   %18.2f%%          %18.2f%%\n' % (cr, ct)
			k = _addLine(scr, k, 0, output, std)
			output = 'Command:           %s' % _getCommandLine(order[sel])
			k = _addLine(scr, k, 0, output[:size[1]], std)
			
			### Clear to the bottom
			scr.clrtobot()
			### Refresh
			scr.refresh()
			
			## Sleep
			time.sleep(_REDRAW_INTERVAL_SEC)
			
	except KeyboardInterrupt:
		pass
		
	curses.nocbreak()
	scr.keypad(0)
	curses.echo()
	curses.endwin()


if __name__ == "__main__":
	main(sys.argv[1:])
	
