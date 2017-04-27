# -*- coding: utf-8 -*-

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

from libbifrost import _bf, _check, _get, _string2space, _space2string
#from GPUArray import GPUArray
from ndarray import ndarray

import ctypes
import numpy as np
from uuid import uuid4

class Ring(object):
	def __init__(self, space='system', name=None):
		if name is None:
			name = str(uuid4())
		space = _string2space(space)
		#self.obj = None
		self.obj = _get(_bf.RingCreate(name=name, space=space), retarg=0)
	def __del__(self):
		if hasattr(self, "obj") and bool(self.obj):
			_bf.RingDestroy(self.obj)
	def resize(self, contiguous_span, total_span=None, nringlet=1,
	           buffer_factor=4):
		if total_span is None:
			total_span = contiguous_span * buffer_factor
		_check( _bf.RingResize(self.obj,
		                       contiguous_span,
		                       total_span,
		                       nringlet) )
	@property
	def name(self):
		return _get(_bf.RingGetName(self.obj))
	@property
	def space(self):
		return _space2string(_get(_bf.RingGetSpace(self.obj)))
	#def begin_sequence(self, name, header="", nringlet=1):
	#	return Sequence(ring=self, name=name, header=header, nringlet=nringlet)
	
	#def end_sequence(self, sequence):
	#	return sequence.end()
	#def _lock(self):
	#	self._check( self.lib.bfRingLock(self.obj) );
	#def unlock(self):
	#	self._check( self.lib.bfRingUnlock(self.obj) );
	#def lock(self):
	#	return RingLock(self)
	def begin_writing(self):
		return RingWriter(self)
	def _begin_writing(self):
		_check( _bf.RingBeginWriting(self.obj) );
	def end_writing(self):
		_check( _bf.RingEndWriting(self.obj) );
	def writing_ended(self):
		return _get( _bf.RingWritingEnded(self.obj) )
	def open_sequence(self, name, guarantee=True):
		return ReadSequence(self, name=name, guarantee=guarantee)
	def open_sequence_at(self, time_tag, guarantee=True):
		return ReadSequence(self, which='at', time_tag=time_tag, guarantee=guarantee)
	def open_latest_sequence(self, guarantee=True):
		return ReadSequence(self, which='latest', guarantee=guarantee)
	def open_earliest_sequence(self, guarantee=True):
		return ReadSequence(self, which='earliest', guarantee=guarantee)
	# TODO: Alternative name?
	def read(self, whence='earliest', guarantee=True):
		with ReadSequence(self, which=whence, guarantee=guarantee) as cur_seq:
			while True:
				yield cur_seq
				cur_seq.increment()
	#def _data(self):
	#	data_ptr = _get(self.lib.bfRingLockedGetData, self.obj)
	#	#data_ptr = c_void_p()
	#	#self._check( self.lib.bfRingLockedGetData(self.obj, pointer(data_ptr)) )
	#	#data_ptr = data_ptr.value
	#	#data_ptr = self.lib.bfRingLockedGetData(self.obj)
	#	if self.space == 'cuda':
	#		# TODO: See if can wrap this in something like PyCUDA's GPUArray
	#		#         Ideally actual GPUArray, but it doesn't appear to support wrapping pointers
	#		return data_ptr
	#	span     = self._total_span()
	#	stride   = self._stride()
	#	nringlet = self._nringlet()
	#	#print "******", span, stride, nringlet
	#	BufferType = c_byte*(nringlet*stride)
	#	data_buffer_ptr = cast(data_ptr, POINTER(BufferType))
	#	data_buffer     = data_buffer_ptr.contents
	#	data_array = np.ndarray(shape=(nringlet, span),
	#	                        strides=(stride, 1) if nringlet > 1 else None,
	#	                        buffer=data_buffer, dtype=np.uint8)
	#	return data_array
	#def _contiguous_span(self):
	#	return self._get(BFsize, self.lib.bfRingLockedGetContiguousSpan, self.obj)
	#def _total_span(self):
	#	return self._get(BFsize, self.lib.bfRingLockedGetTotalSpan, self.obj)
	#def _nringlet(self):
	#	return self._get(BFsize, self.lib.bfRingLockedGetNRinglet, self.obj)
	#def _stride(self):
	#	return self._get(BFsize, self.lib.bfRingLockedGetStride, self.obj)

class RingWriter(object):
	def __init__(self, ring):
		self.ring = ring
		self.ring._begin_writing()
	def __enter__(self):
		return self
	def __exit__(self, type, value, tb):
		self.ring.end_writing()
	def begin_sequence(self, name="", time_tag=-1, header="", nringlet=1):
		return WriteSequence(ring=self.ring, name=name, time_tag=time_tag,
		                     header=header, nringlet=nringlet)

class SequenceBase(object):
        """Python object for a ring's sequence (data unit)"""
	def __init__(self, ring):
		self._ring = ring
	@property
	def _base_obj(self):
		return ctypes.cast(self.obj, _bf.BFsequence)
	@property
	def ring(self):
		return self._ring
	@property
	def name(self):
		return _get(_bf.RingSequenceGetName(self._base_obj))
	@property
	def time_tag(self):
		return _get(_bf.RingSequenceGetTimeTag(self._base_obj))
	@property
	def nringlet(self):
		return _get(_bf.RingSequenceGetNRinglet(self._base_obj))
	@property
	def header_size(self):
		return _get(_bf.RingSequenceGetHeaderSize(self._base_obj))
	@property
	def _header_ptr(self):
		return _get(_bf.RingSequenceGetHeader(self._base_obj))
	@property # TODO: Consider not making this a property
	def header(self):
		size = self.header_size
		if size == 0:
			# WAR for hdr_buffer_ptr.contents crashing when size == 0
			hdr_array = np.empty(0, dtype=np.uint8)
			hdr_array.flags['WRITEABLE'] = False
			return hdr_array
		BufferType = ctypes.c_byte*size
		hdr_buffer_ptr = ctypes.cast(self._header_ptr, ctypes.POINTER(BufferType))
		hdr_buffer = hdr_buffer_ptr.contents
		#hdr_array = memoryview(hdr_buffer)
		# WAR for ctypes producing an invalid type code that numpy fails on
		#hdr_array = buffer(memoryview(hdr_buffer))
		hdr_array = np.frombuffer(hdr_buffer, dtype=np.uint8)
		hdr_array.flags['WRITEABLE'] = False
		return hdr_array

class WriteSequence(SequenceBase):
	def __init__(self, ring, name="", time_tag=-1, header="", nringlet=1):
		SequenceBase.__init__(self, ring)
		# TODO: Allow header to be a string, buffer, or numpy array
		header_size = len(header)
		if isinstance(header, np.ndarray):
			header = header.ctypes.data
		#print "hdr:", header_size, type(header)
		offset_from_head = 0
		self.obj = _get(_bf.RingSequenceBegin(ring=ring.obj,
		                                      name=name,
		                                      time_tag=time_tag,
		                                      header_size=header_size,
		                                      header=header,
		                                      nringlet=nringlet,
		                                      offset_from_head=offset_from_head), retarg=0)
	def __enter__(self):
		return self
	def __exit__(self, type, value, tb):
		self.end()
	def end(self):
		offset_from_head = 0
		_check(_bf.RingSequenceEnd(self.obj, offset_from_head))
	def reserve(self, size):
		return WriteSpan(self.ring, size)

class ReadSequence(SequenceBase):
	def __init__(self, ring, which='specific', name="", time_tag=None,
	             other_obj=None, guarantee=True):
		SequenceBase.__init__(self, ring)
		self._ring = ring
		if which == 'specific':
			self.obj = _get(_bf.RingSequenceOpen(ring=ring.obj,
			                                     name=name, guarantee=guarantee), retarg=0)
		elif which == 'latest':
			self.obj = _get(_bf.RingSequenceOpenLatest(ring=ring.obj,
			                                           guarantee=guarantee), retarg=0)
		elif which == 'earliest':
			self.obj = _get(_bf.RingSequenceOpenEarliest(ring=ring.obj,
			                                             guarantee=guarantee), retarg=0)
		elif which == 'at':
			self.obj = _get(_bf.RingSequenceOpenAt(ring=ring.obj,
			                                       time_tag=time_tag,
			                                       guarantee=guarantee), retarg=0)
		#elif which == 'next':
		#	self._check( self.lib.bfRingSequenceOpenNext(pointer(self.obj), other_obj) )
		else:
			raise ValueError("Invalid 'which' parameter; must be one of: 'specific', 'latest', 'earliest'")
	def __enter__(self):
		return self
	def __exit__(self, type, value, tb):
		self.close()
	def close(self):
		_check(_bf.RingSequenceClose(self.obj))
	#def __next__(self):
	#	return self.next()
	#def next(self):
	#	return ReadSequence(self._ring, which='next', other_obj=self.obj)
	def increment(self):
		#self._check( self.lib.bfRingSequenceNext(pointer(self.obj)) )
		_check(_bf.RingSequenceNext(self.obj))
	def acquire(self, offset, size):
		return ReadSpan(self, offset, size)
	def read(self, span_size, stride=None, begin=0):
		if stride is None:
			stride = span_size
		offset = begin
		while True:
			with self.acquire(offset, span_size) as ispan:
				yield ispan
			offset += stride

class SpanBase(object):
	def __init__(self, ring, writeable):
		self._ring = ring
		self.writeable = writeable
	@property
	def _base_obj(self):
		return ctypes.cast(self.obj, _bf.BFspan)
	@property
	def ring(self):
		return self._ring
	@property
	def size(self):
		return _get(_bf.RingSpanGetSize(self._base_obj))
	@property
	def stride(self):
		return _get(_bf.RingSpanGetStride(self._base_obj))
	@property
	def offset(self):
		return _get(_bf.RingSpanGetOffset(self._base_obj))
	@property
	def nringlet(self):
		return _get(_bf.RingSpanGetNRinglet(self._base_obj))
	@property
	def _data_ptr(self):
		return _get(_bf.RingSpanGetData(self._base_obj))
	@property
	def data(self):
		return self.data_view()
	def data_view(self, dtype=np.uint8, shape=-1):
		itemsize = dtype().itemsize
		assert( self.size   % itemsize == 0 )
		assert( self.stride % itemsize == 0 )
		data_ptr = self._data_ptr
		#if self.sequence.ring.space == 'cuda':
		#	# TODO: See if can wrap this in something like PyCUDA's GPUArray
		#	#         Ideally actual GPUArray, but it doesn't appear to support wrapping pointers
		#	#           Could also try writing a custom GPUArray implem for this purpose
		#	return data_ptr
		span_size  = self.size
		stride     = self.stride
		#nringlet   = self.sequence.nringlet
		nringlet   = self.nringlet
		#print "******", span_size, stride, nringlet
		#BufferType = c_byte*(span_size*self.stride)
		# TODO: We should really map the actual ring memory space and index
		#         it with offset rather than mapping from the current pointer.
		BufferType = ctypes.c_byte*(nringlet*stride)
		data_buffer_ptr = ctypes.cast(data_ptr, ctypes.POINTER(BufferType))
		data_buffer     = data_buffer_ptr.contents
		#print len(data_buffer), (nringlet, span_size), (self.stride, 1)
		_shape   = (nringlet, span_size//itemsize)
		strides = (self.stride, itemsize) if nringlet > 1 else None
		#space   = self.sequence.ring.space
		space   = self.ring.space
		"""
		if space != 'cuda':
			data_array = np.ndarray(shape=_shape, strides=strides,
			                        buffer=data_buffer, dtype=dtype)
		else:
			data_array = GPUArray(shape=_shape, strides=strides,
			                      buffer=data_ptr, dtype=dtype)
			data_array.flags['SPACE'] = space
		"""
		
		data_array = ndarray(shape=_shape, strides=strides,
		                     buffer=data_ptr, dtype=dtype,
		                     space=space)
		
		# Note: This is a non-standard attribute
		#data_array.flags['SPACE'] = space
		if not self.writeable:
			data_array.flags['WRITEABLE'] = False
		if shape != -1:
			# TODO: Check that this still wraps the same memory
			data_array = data_array.reshape(shape)
		return data_array
	#@property
	#def sequence(self):
	#	return self._sequence

class WriteSpan(SpanBase):
	def __init__(self,
	             ring,
	             size):
		SpanBase.__init__(self, ring, writeable=True)
		self.obj = _get(_bf.RingSpanReserve(ring=ring.obj, size=size), retarg=0)
		self.commit_size = size
	def commit(self, size):
		self.commit_size = size
	def __enter__(self):
		return self
	def __exit__(self, type, value, tb):
		self.close()
	def close(self):
		_check(_bf.RingSpanCommit(self.obj, self.commit_size))

class ReadSpan(SpanBase):
	def __init__(self, sequence, offset, size):
		SpanBase.__init__(self, sequence.ring, writeable=False)
		self.obj = _get(_bf.RingSpanAcquire(sequence=sequence.obj,
		                                    offset=offset, size=size), retarg=0)
	def __enter__(self):
		return self
	def __exit__(self, type, value, tb):
		self.release()
	def release(self):
		_check(_bf.RingSpanRelease(self.obj))
