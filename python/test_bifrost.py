#!/usr/bin/env python

# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

