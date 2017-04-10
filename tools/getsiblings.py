#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import getopt


def usage(exitCode=None):
	print """%s - Get sibling cores on HT systems

Usage: %s [OPTIONS] [core [core [...]]]

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


def _readSiblings():
	cpus = glob.glob('/sys/devices/system/cpu/cpu*/topology/thread_siblings_list')
	
	siblings = {}
	for cpu in cpus:
		cid = cpu.split('/topology', 1)[0]
		cid = cid.rsplit('cpu', 1)[1]
		cid = int(cid, 10)
		
		fh = open(cpu)
		data = fh.read()
		fh.close()
		
		siblings[cid] = [int(v,10) for v in data.split(',')]
		
	return siblings


def main(args):
	config = parseOptions(args)
	
	siblings = _readSiblings()
	
	if len(config['args']) == 0:
		for cpu in sorted(siblings.keys()):
			print "%i: %s" % (cpu, str(siblings[cpu]))
	else:
		for cpu in config['args']:
			cpu = int(cpu, 10)
			try:
				data = str(siblings[cpu])
			except KeyError:
				data = "not found"
			print "%i: %s" % (cpu, str(siblings[cpu]))


if __name__ == '__main__':
	main(sys.argv[1:])
	