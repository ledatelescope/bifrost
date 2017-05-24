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
from datetime import datetime

os.environ['VMA_TRACELEVEL'] = '0'
from bifrost.proclog import load_by_pid


BIFROST_STATS_BASE_DIR = '/dev/shm/bifrost/'

def usage(exitCode=None):
	print """%s - Write logging information for a bifrost file to a flat text file for analysis

Usage: %s [OPTIONS] PID

Options:
-h, --help                  Display this help information
-i, --interval              Logging interval in seconds (default = 1)
-o, --output                Write the log to specified file (default = show)
-r, --rx                    Log network receive statistics (default if no other
                            fields specified)
-t, --tx                    Log network transmit statistics
-b, --block-timing          Log per-block timing statistics
-c, --block-cpu             Log per-block CPU usage
""" % (os.path.basename(__file__), os.path.basename(__file__))
	
	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseOptions(args):
	config = {}
	# Command line flags - default values
	config['args'] = []
	config['interval'] = 1.0
	config['output'] = None
	config['mode'] = []
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hi:o:rtbc", ["help", "interval=", "output=", "rx", "tx", "block-timing", "block-cpu"])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-i', '--interval'):
			config['interval'] = max([1.0, float(value)])
		elif opt in ('-o', '--output'):
			config['output'] = value
		elif opt in ('-r', '--rx'):
			config['mode'].append( 'rx' )
		elif opt in ('-t', '--tx'):
			config['mode'].append( 'tx' )
		elif opt in ('-b', '--block-timing'):
			config['mode'].append( 'timing' )
		elif opt in ('-c', '--block-cpu'):
			config['mode'].append( 'cpu' )
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Validate
	if len(config['mode']) == 0:
		config['mode'].append( 'rx' )
	config['mode'] = list(set(config['mode']))
	if len(config['args']) != 1:
		raise RuntimeError("Must specify one and only one PID to log")
		
	# Return configuration
	return config


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


def _getTransmitReceive(pid):
	"""
	Read in the /dev/bifrost ProcLog data and return block-level information 
	about udp* blocks.
	"""
	
	## Find all running processes
	pidDirs = [os.path.join(BIFROST_STATS_BASE_DIR, str(pid)),]
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
				
			blockList[block] = {'pid': pid, 'name':block, 
							'time':t, 
							'good': good, 'missing': missing, 
							'invalid': invalid, 'late': late, 
							'nvalid': nvalid}
	return blockList


def _getBlockList(pid):
	"""
	Read in the /dev/bifrost ProcLog data and return block-level information 
	about all blocks.
	"""
	
	## Find all running processes
	pidDirs = [os.path.join(BIFROST_STATS_BASE_DIR, str(pid)),]
	pidDirs.sort()
	
	## Load the data
	blockList = {}
	for pidDir in pidDirs:
		pid = int(os.path.basename(pidDir), 10)
		contents = load_by_pid(pid)
		
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
			
			blockList[block] = {'name':block, 
							'core': cr, 
							'acquire': ac, 
							'process': pr, 
							'reserve': re, 
							'total':ac+pr+re}
							
	return blockList


def _getStatistics(blockList, prevList):
	"""
	Given a list of running blocks and a previous version of that, compute 
	basic statistics for the UDP blocks.
	"""
	
	# Loop over the blocks to find udp_capture and udp_transmit blocks
	output = {'updated': time.time()}
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
			#output[pid]['cmd'] = _getCommandLine(pid)
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


def main(args):
	config = parseOptions(args)
	pid = int(config['args'][0], 10)
	
	# Initial polling to get setup
	cpu  = _getProcessorUsage()
	blocks  = _getBlockList(pid)
	network = _getTransmitReceive(pid)
	time.sleep(config['interval'])
	
	# Figure out what we should log
	## Build
	fields, labels = [], []
	if 'rx' in config['mode']:
		fields.extend([(pid,'rx','good'), 
					(pid,'rx','missing'), 
					(pid,'rx','invalid'), 
					(pid,'rx','late'), 
					(pid,'rx','drate'), 
					(pid,'rx','prate'), 
					(pid,'rx','gloss'), 
					(pid,'rx','closs'),])
		labels.extend(['RX Good [B]', 
					'RX Missing [B]', 
					'RX Invalid [B]', 
					'RX Late [B]', 
					'RX Data Rate [B/s]', 
					'RX Packet Rate [pkt/s]', 
					'RX Global Packet Loss [%]', 
					'RX Current Packet Loss [%]',])
	if 'tx' in config['mode']:
		fields.extend([(pid,'tx','good'), 
					(pid,'tx','drate'), 
					(pid,'tx','prate'),])
		labels.extend(['TX Good [B]', 
					'TX Data Rate [B/s]',
					'TX Packet Rate [pkt/s]',])
	if 'timing' in config['mode']:
		for block in sorted(blocks.keys()):
			for f in ('acquire', 'process', 'reserve', 'total'):
				fields.append( (block,f) )
				labels.append('%s - %s Time [s]' % (block, f.capitalize()))
	if 'cpu' in config['mode']:
		for block in sorted(blocks.keys()):
			fields.append( (block,'core') )
			labels.append('%s - CPU Usage [%%]' % block)
			
	## Report
	if config['output'] is None:
		out = sys.stdout
	else:
		out = open(config['output'], 'w')
	out.write("# Date: %s\n" % datetime.utcnow())
	out.write("#\n")
	out.write("# bifrost process being logged: %i\n" % pid)
	out.write("# Process command line: %s\n" % _getCommandLine(pid))
	out.write("# Number of fields logged: %i\n" % (len(fields)+1,))
	out.write("# Logging interval: %.1f s\n" % config['interval'])
	out.write("#\n")
	out.write("# Columns:\n")
	out.write("#   (%2i) Timestamp [s]\n" % 1)
	for i,l in enumerate(labels):
		out.write("#   (%2i) %s\n" % (i+2, l))
		
	# Main logging loop
	try:
		while True:
			## Save what we had before for network statistics
			prevNetwork = network
			
			## Refresh the data
			cpu  = _getProcessorUsage()
			blocks  = _getBlockList(pid)
			network = _getTransmitReceive(pid)
			if len(blocks) == 0:
				### Everything is gone, giving up
				break
				
			## Update the network stats
			stats = _getStatistics(network, prevNetwork)
			
			## Report
			output = ["%.4f" % stats['updated'],]
			for f in fields:
				if len(f) == 3:
					### Network fields have three parts
					output.append( "%.4f" % stats[f[0]][f[1]][f[2]] )
				elif len(f) == 2:
					### Block fields have only two
					if f[1] == 'core':
						## Special catch to make sure we pull out the CPU correctly
						c = 100.0*cpu[blocks[f[0]][f[1]]]['total']
						output.append("%.1f" % c)
					else:
						output.append( "%.4f" % blocks[f[0]][f[1]] )
				else:
					pass
			output = ",".join(output)
			out.write("%s\n" % output)
			
			## Sleep
			time.sleep(config['interval'])
			
	except KeyboardInterrupt:
		pass
		
	if config['output'] is not None:
		out.close()
		print "Wrote %.0f kB to '%s'" % (os.path.getsize(config['output'])/1024., config['output'])


if __name__ == "__main__":
	main(sys.argv[1:])
	