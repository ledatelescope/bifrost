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


def _getProcessDetails(pid):
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
	cmd = ''
	
	try:
		with open('/proc/%i/cmdline' % pid, 'r') as fh:
			cmd = fh.read()
			cmd = cmd.replace('\0', ' ')
			fh.close()
	except IOError:
		pass
	return cmd


def main(args):
	config = parseOptions(args)
	
	pidDirs = glob.glob(os.path.join(BIFROST_STATS_BASE_DIR, '*'))
	pidDirs.sort()
	
	for pidDir in pidDirs:
		pid = int(os.path.basename(pidDir), 10)
		contents = load_by_pid(pid)
		
		details = _getProcessDetails(pid)
		cmd = _getCommandLine(pid)
		
		if cmd == '' and details['user'] == '':
			continue
			
		print "PID: %i" % pid
		print "  Command: %s" % cmd
		print "  User: %s" % details['user']
		print "  CPU Usage: %.1f%%" % details['cpu']
		print "  Memory Usage: %.1f%%" % details['mem']
		print "  Elapsed Time: %s" % details['etime']
		print "  Thread Count: %i" % details['threads']
		print "  Rings:"
		rings = []
		for block in contents.keys():
			for log in contents[block].keys():
				if log not in ('in', 'out'):
					continue
				for key in contents[block][log]:
					if key[:4] == 'ring':
						value = contents[block][log][key]
						if value not in rings:
							rings.append( value )
		for i,ring in enumerate(rings):
			print "    %i: %s" % (i, ring)
		print "  Blocks:"
		for block in contents.keys():
			rins, routs = [], []
			for log in contents[block].keys():
				if log not in ('in', 'out'):
					continue
				for key in contents[block][log]:
					if key[:4] == 'ring':
						value = contents[block][log][key]
						if log == 'in':
							if value not in rins:
								rins.append( value )
						else:
							if value not in rins:
								routs.append( value )
			print "    %s" % block
			if len(rins) > 0:
				print "      -> read ring(s): %s" % (" ".join(["%i" % rings.index(v) for v in rins]),)
			if len(routs) > 0:
				print "      -> write ring(s): %s" % (" ".join(["%i" % rings.index(v) for v in routs]),)
			if len(contents[block].keys()) > 0:
				print "      -> log(s): %s" % (" ".join(contents[block].keys()),)


if __name__ == "__main__":
	main(sys.argv[1:])
	
