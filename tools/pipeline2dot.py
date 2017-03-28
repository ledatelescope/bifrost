#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import getopt
import subprocess

from bifrost.proclog import load_by_pid


BIFROST_STATS_BASE_DIR = '/dev/shm/bifrost/'

def usage(exitCode=None):
	print """%s - Display details of running bifrost processes

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
	
	# Validate
	if len(config['args']) != 1:
		raise RuntimeError("Need to specify a PID to diagram")
	
	# Return configuration
	return config


def _getProcessDetails(pid):
	"""
	Use a call to 'ps' to get details about the specified PID.  These details
	include:
	  * the user running the process, 
	  * the CPU usage of the process, 
	  * the memory usage of the process, 
	  * the process elapsed time, and
	  * the number of threads.
	These are returned as a dictionary.
	
	NOTE::  Using 'ps' to get this is slightly easier than directly querying 
		  /proc, altough that method might be preferred.
		  
	NOTE::  Many of these details could be avoided by using something like the
		  Python 'psutil' module.
	"""
	
	data = {'user':'', 'cpu':0.0, 'mem':0.0, 'etime':'00:00', 'threads':0}
	try:
		output = subprocess.check_output('ps o user,pcpu,pmem,etime,nlwp %i' % pid, shell=True)
		output = output.split('\n')[1]
		fields = output.split(None, 4)
		data['user'] = fields[0]
		data['cpu'] = float(fields[1])
		data['mem'] = float(fields[2])
		data['etime'] = fields[3].replace('-', 'd ')
		data['threads'] = int(fields[4], 10)
	except subprocess.CalledProcessError:
		pass
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


def _getDataFlows(blocks):
	"""
	Given a block dictonary from bifrost.proclog.load_by_pid(), return a list
	of chains that give the data flow.
	"""
	
	# Find out what rings we have to work with and which blocks are sources 
	# or sinks
	rings = []
	sources, sourceRings = [], []
	sinks, sinkRings = [], []
	for block in blocks.keys():
		rins, routs = [], []
		rFound = False
		for log in blocks[block].keys():
			if log not in ('in', 'out'):
				continue
			for key in blocks[block][log]:
				if key[:4] == 'ring':
					rFound = True
					
					value = blocks[block][log][key]
					if value not in rings:
						rings.append( value )
					if log == 'in':
						if value not in rins:
							rins.append( value )
					else:
						if value not in routs:
							routs.append( value )
							
		if rFound:
			if len(rins) == 0:
				sources.append( block )
				sourceRings.extend( routs )
			if len(routs) == 0:
				sinks.append( block )
				sinkRings.extend( rins )
				
	# Find out the chains
	chains = []
	for refRing in rings:
		for block in blocks.keys():
			rins, routs = [], []
			for log in blocks[block].keys():
				if log not in ('in', 'out'):
					continue
				for key in blocks[block][log]:
					if key[:4] == 'ring':
						value = blocks[block][log][key]
						if log == 'in':
							if value not in rins:
								rins.append( value )
						else:
							if value not in routs:
								routs.append( value )
			if refRing in routs:
				refBlock = block
				refROuts = routs
				
				for block in blocks.keys():
					rins, routs = [], []
					dtype = None
					for log in blocks[block].keys():
						if log.startswith('sequence'):
							try:
								bits = blocks[block][log]['nbit']
								if blocks[block][log]['complex']:
									bits *= 2
								name = 'comp' if  blocks[block][log]['complex'] else 'real'
								dtype = '%s%i' % (name, bits)
							except KeyError:
								pass
						elif log not in ('in', 'out'):
							continue
							
						for key in blocks[block][log]:
							if key[:4] == 'ring':
								value = blocks[block][log][key]
								if log == 'in':
									if value not in rins:
										rins.append( value )
								else:
									if value not in routs:
										routs.append( value )
										
					for ring in rins:
						if ring in refROuts:
							#print refRing, rins, block
							chains.append( {'link':(refBlock,block), 'dtype':dtype} )
	return sources, sinks, chains


def main(args):
	config = parseOptions(args)
	pidToUse = int(config['args'][0], 10)
	
	pidDirs = glob.glob(os.path.join(BIFROST_STATS_BASE_DIR, '*'))
	pidDirs.sort()
	
	for pidDir in pidDirs:
		pid = int(os.path.basename(pidDir), 10)
		if pid != pidToUse:
			continue
			
		contents = load_by_pid(pid)
		
		details = _getProcessDetails(pid)
		cmd = _getCommandLine(pid)
		
		if cmd == '' and details['user'] == '':
			continue
			
		# Assign unique one-character IDs to each block
		lut = {}
		for i,block in enumerate(contents.keys()):
			lut[block] = chr(i+97)
			
		# Find chains of linked blocks
		sources, sinks, chains = _getDataFlows(contents)
		
		# Trim the command line
		if cmd.startswith('python'):
			cmd = cmd.split(None, 1)[1]
		cmd = cmd.split(None, 1)[0]
		cmd = os.path.basename(cmd)
		
		# Create the DOT output
		print "digraph graph%i {" % pid
		## Graph label
		print '  labelloc="t"'
		print '  label="Pipeline: %s\\n "' % cmd
		## Block identiers
		for block in sorted(lut):
			### Is the block actually used?
			found = False
			for chain in chains:
				for link in chain['link']:
					if link == block:
						found = True
						break
				if found:
					break
					
			if found:
				### Yes, add it to the graph with the correct label
				## CPU info - if avaliable
				try:
					cpu = contents[block]['bind']['core0']
					cpu = 'CPU%i' % cpu
				except KeyError:
					cpu = "Unbound"
				## Shape - based on function (source vs. sink vs. connection)
				shape = 'box'
				if block in sources:
					shape = 'ellipse'
				if block in sinks:
					shape = 'diamond'
				## Add it to the list
				print '  %s [label="%s\\n%s" shape="%s"]' % (lut[block], block, cpu, shape)
				
		## Chains
		for chain in chains:
			### Extract the data type, if known
			dtype = chain['dtype']
			if dtype is None:
				dtype = ''
			else:
				dtype = ' %s' % dtype
			### Add it to the list
			print '  %s -> %s [label="%s"]' % (lut[chain['link'][0]], lut[chain['link'][1]], dtype)
		print "}"


if __name__ == "__main__":
	main(sys.argv[1:])
	
