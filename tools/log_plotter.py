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
import numpy
import getopt

from matplotlib import pyplot as plt


def usage(exitCode=None):
	print """%s - Plot logging information from a bifrost log text file

Usage: %s [OPTIONS] filename

Options:
-h, --help                  Display this help information
-c, --columns               Comma-separated list of column numbers to plot
                            (default = prompt for columns to plot)
""" % (os.path.basename(__file__), os.path.basename(__file__))
	
	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseOptions(args):
	config = {}
	# Command line flags - default values
	config['args'] = []
	config['cols'] = ''
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hc:", ["help", "columns="])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-c', '--columns'):
			config['cols'] = value
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Validate
	if len(config['args']) != 1:
		raise RuntimeError("Must specify one and only one log to plot")
		
	# Return configuration
	return config


def _parseHeader(filename):
	fh = open(filename, 'r')
	
	metadata = []
	fields = {}
	for line in fh:
		line = line.replace('\n', '')
		if len(line) < 3:
			continue
		if line[0] != '#':
			break
			
		if line.find('(') != -1 and line.find(')') != -1:
			n, l = line.split(')', 1)
			n = n.split('(', 1)[1]
			n = int(n, 10) - 1
			l = l.strip().rstrip()
			fields[n] = l
		else:
			m = line.replace('#', '')
			m = m.strip().rstrip()
			metadata.append(m)
	fh.close()
	
	return metadata, fields


def main(args):
	# Parse the command line
	config = parseOptions(args)
	filename = config['args'][0]
	
	# Figure out what to plot
	## Query
	metadata, fields = _parseHeader(filename)
	## Report
	print "Log Information:"
	for m in metadata:
		print "  %s" % m
	print "Fields:"
	for n in sorted(fields.keys()):
		if fields[n].find('Timestamp') != -1:
			toPlotX = n
			continue
		print "  %i - %s" % (n, fields[n])
	## Ask
	print "Enter a comma-separated list of fields to plot or 'q' to quit:"
	if config['cols'] == '':
		raw = raw_input("$ ")
	else:
		print "$ %s" % config['cols']
		raw = config['cols']
	## Parse
	### Exit?
	if len(raw) == 0:
		sys.exit()
	elif raw.find('q') != -1:
		sys.exit()
	### Really parse
	toPlotY = []
	for v in raw.split(','):
		if v.find('-') != -1:
			try:
				f,s = v.split('-', 1)
				f,s = int(f,10),int(s,10)
				v = range(f, s+1)
			except KeyError:
				v = [int(v, 10),]
		else:
			v = [int(v, 10),]
		toPlotY.extend(v)
	toPlotY = list(set(toPlotY))
	
	# Group things together
	groups = {}
	for n in toPlotY:
		label = fields[n]
		unit = label.split('[', 1)[1]
		unit = unit.split(']', 1)[0]
		unit = unit.strip().rstrip()
		if label.find('RX') != -1 or label.find('TX') != -1:
			group = 'network-%s' % unit
		else:
			group = unit
			
		try:
			groups[group].append( n )
		except KeyError:
			groups[group] = [n,]
			
	# Load the data
	data = numpy.loadtxt(filename, comments='#', delimiter=',')
	tElapsed = data[:,toPlotX] - data[0,toPlotX]
	
	# Plot
	for group in groups:
		## Skip over empty groups
		if len(groups[group]) == 0:
			continue
			
		## Setup the figure
		fig = plt.figure()
		ax = fig.gca()
		
		## Label the figure as needed
		ax.set_xlabel('Elapsed Time [s]')
		unit = group.replace('network-', '')
		norm = 1.0
		if unit == 'B':
			scale = data[:,groups[group]].mean()
			if scale >= 1024**4:
				norm = 1024.0**4
				unit = 'T%s' % unit
			elif scale >= 1024**3:
				norm = 1024.0**3
				unit = 'G%s' % unit
			elif scale >= 1024**2:
				norm = 1024.0**2
				unit = 'M%s' % unit
			elif scale >= 1024:
				norm = 1024.0
				unit = 'k%s' % unit
			label = 'Data [%s]' % unit
		elif unit == 'B/s':
			scale = data[:,groups[group]].mean()
			if scale >= 1024**4:
				norm = 1024.0**4
				unit = 'T%s' % unit
			elif scale >= 1024**3:
				norm = 1024.0**3
				unit = 'G%s' % unit
			elif scale >= 1024**2:
				norm = 1024.0**2
				unit = 'M%s' % unit
			elif scale >= 1024:
				norm = 1024.0
				unit = 'k%s' % unit
			label = 'Data Rate [%s]' % unit
		elif unit == 'pkt/s':
			scale =  data[:,groups[group]].mean()
			if scale >= 1000:
				norm = 1000.0
				unit = 'k%s' % unit
			label = 'Packet Rate [%s]' % unit
		elif unit == 's':
			label = 'Time [%s]' % unit
		elif unit == '%' and group.find('network') != -1:
			label = 'Packet Loss [%s]' % unit
		else:
			label = 'CPU Usage [%]'
		ax.set_ylabel(label)
		ax.set_title(' '.join(label.rsplit(None)[:-1]))
		
		## Plot the figure, labeling the lines as needed
		index = []
		counts = {}
		colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
		linestyles = ['-', '--', '-.', '-']
		markers = ['+', '+', '+', 'x']
		for n in groups[group]:
			### Come up with a label
			label = fields[n]
			label = label.rsplit('[', 1)[0]
			label = label.replace('Packet Loss', '')
			label = label.replace('Data Rate', '')
			label = label.replace('Packet Rate', '')
			label = label.replace('Time', '')
			if unit == '%':
				label = label.split('-', 1)[0]
			label = label.strip().rstrip()
			
			### Figure out the sub-grouping
			try:
				subgroup, junk = label.split('-', 1)
				try:
					counts[subgroup] += 1
				except KeyError:
					index.append( subgroup )
					counts[subgroup] = 0
				color = colors[index.index(subgroup)]
				linestyle = linestyles[counts[subgroup]%len(linestyles)]
				marker = markers[counts[subgroup]%len(markers)]
			except ValueError:
				try:
					del color
				except NameError:
					pass
				linestyle='-'
				marker = '+'
				
			### Add the line
			try:
				ax.plot(tElapsed, data[:,n]/norm, linestyle=linestyle, marker=marker, color=color, label=label)
			except NameError:
				ax.plot(tElapsed, data[:,n]/norm, linestyle=linestyle, marker=marker, label=label)
			
		## Done with this figure
		ax.legend(loc=0)
		plt.draw()
	plt.show()


if __name__ == "__main__":
	main(sys.argv[1:])
	