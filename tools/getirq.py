#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import getopt


def usage(exitCode=None):
	print """%s - List the IRQ bindings for a particular network interface

Usage: %s [OPTIONS] interface

Options:
-h, --help             Display this help information
""" % (os.path.basname(__file__), os.path.basename(__file__))
	
	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseConfig(args):
	config = {}
	config['args'] = []
	
	# Read in and process the command line flags
	try:
		opts, arg = getopt.getopt(args, "h", ["help",])
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
	config['args'] = arg
	
	# Validate
	if len(config['args']) != 1:
		raise RuntimeError("Need to specify a device name")
		
	# Return configuration
	return config


def main(args):
	config = parseConfig(args)
	interface = config['args'][0]
	
	fh = open('/proc/interrupts', 'r')
	lines = fh.read()
	fh.close()
	
	irqs = {}
	for line in lines.split('\n'):
		if line.find(interface) != -1:
			fields = line.split()
			irq = int(fields[0][:-1], 10)
			procs = [int(v,10) for v in fields[1:-2]]
			type = fields[-2]
			name = fields[-1]
			
			mv = max(procs)
			mi = procs.index(mv)
			irqs[irq] = {'cpu':mi, 'type':type, 'name':name, 'count':mv}
	total = sum([irqs[irq]['count'] for irq in irqs])
	
	print "Interface: %s" % interface
	print "%4s  %16s  %16s  %4s  %6s" % ('IRQ', 'Name', 'Type', 'CPU', 'Usage')  
	for irq in sorted(irqs.keys()):
		print "%4i  %16s  %16s  %4i  %5.1f%%" % (irq, irqs[irq]['name'], irqs[irq]['type'], irqs[irq]['cpu'], 100.0*irqs[irq]['count']/total)
 

if __name__ == "__main__":
	main(sys.argv[1:])
	
