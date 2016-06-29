#!/usr/bin/env python

#  Copyright 2016 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import bifrost
from bifrost.ring import Ring
from bifrost import GPUArray

from sigproc import SigprocFile

import numpy as np
import threading
import json

def test_GPUArray():
	np.random.seed(1234)
	a = GPUArray(shape=(3,4), dtype=np.float32)
	print a.shape
	r = np.random.random(size=(3,4))
	print r
	a.set(np.arange(3*4).reshape(3,4))
	print a.get()

class SigprocReadOp(object):
	def __init__(self, filenames, outring, gulp_nframe=4096, core=-1):
		self.filenames   = filenames
		self.oring       = outring
		self.gulp_nframe = gulp_nframe
		self.core        = core
	def main(self): # Launched in thread
		bifrost.affinity.set_core(self.core)
		with self.oring.begin_writing() as oring:
			for name in self.filenames:
				print "Opening", name
				with SigprocFile().open(name,'rb') as ifile:
                                        ifile.read_header()
					ohdr = {}
					ohdr['frame_shape']   = (ifile.nchans, ifile.nifs)
					ohdr['frame_size']    = ifile.nchans*ifile.nifs
					ohdr['frame_nbyte']   = ifile.nchans*ifile.nifs*ifile.nbits/8
					ohdr['frame_axes']    = ('pol', 'chan')
					ohdr['ringlet_shape'] = (1,)
					ohdr['ringlet_axes']  = ()
					ohdr['dtype']         = str(ifile.dtype)
					ohdr['nbit']          = ifile.nbits
					print 'ohdr:', ohdr
					ohdr = json.dumps(ohdr)
                                        gulp_nbyte = self.gulp_nframe*ifile.nchans*ifile.nifs*ifile.nbits/8
					self.oring.resize(gulp_nbyte)
					with oring.begin_sequence(name, header=ohdr) as osequence:
						while True:
							with osequence.reserve(gulp_nbyte) as wspan:
								size = ifile.file_object.readinto(wspan.data.data)
								wspan.commit(size)
								#print size
								if size == 0:
									break

class CopyOp(object):
	def __init__(self, iring, oring, gulp_size=1048576, guarantee=True, core=-1):
		self.gulp_size  = gulp_size
		self.iring      = iring
		self.oring      = oring
		self.guarantee  = guarantee
		self.core       = core
	def main(self):
		bifrost.affinity.set_core(self.core)
		self.iring.resize(self.gulp_size)
		self.oring.resize(self.gulp_size)
		with self.oring.begin_writing() as oring:
			for iseq in self.iring.read(guarantee=self.guarantee):
				with oring.begin_sequence(iseq.name, iseq.time_tag,
				                          header=iseq.header,
				                          nringlet=iseq.nringlet) as oseq:
					for ispan in iseq.read(self.gulp_size):
						with oseq.reserve(ispan.size) as ospan:
							bifrost.memory.memcpy2D(ospan.data,
							                        ispan.data)
							#print "Copied", ospan.size

class PrintOp(object):
	def __init__(self, iring, gulp_size=1048576, guarantee=True):
		self.gulp_size  = gulp_size
		self.iring      = iring
		self.guarantee  = guarantee
	def main(self):
		self.iring.resize(self.gulp_size)
		for iseq in self.iring.read(guarantee=self.guarantee):
			print iseq.name, iseq.header.tostring()
			for ispan in iseq.read(self.gulp_size):
				print ispan.size, ispan.offset
				#print ispan.data.shape, ispan.data.dtype
				#pass

def main():
	test_GPUArray()
	
	filenames = ['/data1/mcranmer/data/fake/pulsar.fil']
	
	ring0 = Ring()
	ring1 = Ring()
	
	ops = []
	ops.append(SigprocReadOp(filenames, ring0, core=0))
	ops.append(CopyOp(ring0, ring1, core=1))
	ops.append(PrintOp(ring1))
	
	threads = [threading.Thread(target=op.main) for op in ops]
	print "Launching %i threads" % len(threads)
	for thread in threads:
		thread.daemon = True
		thread.start()
	print "Waiting for threads to finish"
	#while not shutdown_event.is_set():
	#	signal.pause()
	for thread in threads:
		thread.join()
	print "Done"
	"""
	ring = Ring()
	print ring.space
	ring.resize(4)
	with ring.begin_writing() as wring:
		with wring.begin_sequence() as wseq:
			for i in xrange(10):
				with wseq.reserve(4) as wspan:
					data = wspan.data_view(dtype=np.float32)
					print data.dtype, data.shape
					data[...] = 7
	"""

if __name__ == '__main__':
   main()

