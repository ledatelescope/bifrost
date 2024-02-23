
# Copyright (c) 2016-2023, The Bifrost Authors. All rights reserved.
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

from bifrost.libbifrost import _bf, _check, _get, BifrostObject, _string2space, _space2string, EndOfDataStop
from bifrost.DataType import DataType
from bifrost.ndarray import ndarray, _address_as_buffer

import ctypes
import string
import numpy as np
from uuid import uuid4

from typing import List, Optional, Tuple, Union

from bifrost import telemetry
telemetry.track_module()

def _slugify(name):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    valid_chars = frozenset(valid_chars)
    return ''.join([c for c in name if c in valid_chars])

class Ring(BifrostObject):
    def __init__(self, space: str='system', name: Optional[str]=None, core: Optional[int]=None):
        if name is None:
            name = str(uuid4())
        name = _slugify(name)
        space = _string2space(space)
        #self.obj = None
        #self.obj = _get(_bf.bfRingCreate(name=name, space=space), retarg=0)
        BifrostObject.__init__(
            self, _bf.bfRingCreate, _bf.bfRingDestroy, name.encode(), space)
        if core is not None:
            try:
                _check( _bf.bfRingSetAffinity(self.obj, 
                                              core) )
            except RuntimeError:
                pass
    def resize(self, contiguous_span: int, total_span: Optional[int]=None, nringlet: int=1,
               buffer_factor: int=4) -> None:
        if total_span is None:
            total_span = contiguous_span * buffer_factor
        _check( _bf.bfRingResize(self.obj,
                                 contiguous_span,
                                 total_span,
                                 nringlet) )
    @property
    def name(self) -> str:
        n = _get(_bf.bfRingGetName, self.obj)
        return n.decode()
    @property
    def space(self) -> str:
        return _space2string(_get(_bf.bfRingGetSpace, self.obj))
    @property
    def core(self) -> int:
        return _get(_bf.bfRingGetAffinity, self.obj)
    #def begin_sequence(self, name, header="", nringlet=1):
    #    return Sequence(ring=self, name=name, header=header, nringlet=nringlet)
    #def end_sequence(self, sequence):
    #    return sequence.end()
    #def _lock(self):
    #    self._check( self.lib.bfRingLock(self.obj) );
    #def unlock(self):
    #    self._check( self.lib.bfRingUnlock(self.obj) );
    #def lock(self):
    #    return RingLock(self)
    def begin_writing(self) -> "RingWriter":
        return RingWriter(self)
    def _begin_writing(self):
        _check( _bf.bfRingBeginWriting(self.obj) )
    def end_writing(self) -> None:
        _check( _bf.bfRingEndWriting(self.obj) )
    def writing_ended(self) -> bool:
        return _get( _bf.bfRingWritingEnded, self.obj )
    def open_sequence(self, name: str, guarantee: bool=True) -> "ReadSequence":
        return ReadSequence(self, name=name, guarantee=guarantee)
    def open_sequence_at(self, time_tag: int, guarantee: bool=True) -> "ReadSequence":
        return ReadSequence(self, which='at', time_tag=time_tag, guarantee=guarantee)
    def open_latest_sequence(self, guarantee: bool=True) -> "ReadSequence":
        return ReadSequence(self, which='latest', guarantee=guarantee)
    def open_earliest_sequence(self, guarantee: bool=True) -> "ReadSequence":
        return ReadSequence(self, which='earliest', guarantee=guarantee)
    # TODO: Alternative name?
    def read(self, whence: str='earliest', guarantee: bool=True) -> "ReadSequence":
        with ReadSequence(self, which=whence, guarantee=guarantee) as cur_seq:
            while True:
                try:
                    yield cur_seq
                    cur_seq.increment()
                except EndOfDataStop:
                    return

class RingWriter(object):
    def __init__(self, ring: Ring):
        self.ring = ring
        self.ring._begin_writing()
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.ring.end_writing()
    def begin_sequence(self, name: str="", time_tag: int=-1,
                       header: str="", nringlet: int=1) -> "WriteSequence":
        return WriteSequence(ring=self.ring, name=name, time_tag=time_tag,
                             header=header, nringlet=nringlet)

class SequenceBase(object):
    """Python object for a ring's sequence (data unit)"""
    def __init__(self, ring: Ring):
        self._ring = ring
    @property
    def _base_obj(self):
        return ctypes.cast(self.obj, _bf.BFsequence)
    @property
    def ring(self) -> Ring:
        return self._ring
    @property
    def name(self) -> str:
        n = _get(_bf.bfRingSequenceGetName, self._base_obj)
        return n.decode()
    @property
    def time_tag(self) -> int:
        return _get(_bf.bfRingSequenceGetTimeTag, self._base_obj)
    @property
    def nringlet(self) -> int:
        return _get(_bf.bfRingSequenceGetNRinglet, self._base_obj)
    @property
    def header_size(self) -> int:
        return _get(_bf.bfRingSequenceGetHeaderSize, self._base_obj)
    @property
    def _header_ptr(self):
        return _get(_bf.bfRingSequenceGetHeader, self._base_obj)
    @property # TODO: Consider not making this a property
    def header(self) -> np.ndarray:
        size = self.header_size
        if size == 0:
            # WAR for hdr_buffer_ptr.contents crashing when size == 0
            hdr_array = np.empty(0, dtype=np.uint8)
            hdr_array.flags['WRITEABLE'] = False
            return hdr_array
        hdr_buffer = _address_as_buffer(self._header_ptr, size, readonly=True)
        hdr_array = np.frombuffer(hdr_buffer, dtype=np.uint8)
        hdr_array.flags['WRITEABLE'] = False
        return hdr_array

class WriteSequence(SequenceBase):
    def __init__(self, ring: Ring, name: str="", time_tag: int=-1, header: str="", nringlet: int=1):
        SequenceBase.__init__(self, ring)
        # TODO: Allow header to be a string, buffer, or numpy array
        header_size = len(header)
        if isinstance(header, np.ndarray):
            header = header.ctypes.data
        elif isinstance(header, str):
            header = header.encode()
        #print("hdr:", header_size, type(header))
        name = str(name)
        offset_from_head = 0
        self.obj = _bf.BFwsequence()
        _check(_bf.bfRingSequenceBegin(
            self.obj,
            ring.obj,
            name.encode(),
            time_tag,
            header_size,
            header,
            nringlet,
            offset_from_head))
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.end()
    def end(self) -> None:
        offset_from_head = 0
        _check(_bf.bfRingSequenceEnd(self.obj, offset_from_head))
    def reserve(self, size: int, nonblocking: bool=False) -> "WriteSpan":
        return WriteSpan(self.ring, size, nonblocking)

class ReadSequence(SequenceBase):
    def __init__(self, ring: Ring, which: str='specific', name: str="",
                 time_tag: Optional[int]=None, other_obj: Optional[SequenceBase]=None,
                 guarantee: bool=True):
        SequenceBase.__init__(self, ring)
        self._ring = ring
        self.obj = _bf.BFrsequence()
        if which == 'specific':
            _check(_bf.bfRingSequenceOpen(self.obj, ring.obj, name, guarantee))
        elif which == 'latest':
            _check(_bf.bfRingSequenceOpenLatest(self.obj, ring.obj, guarantee))
        elif which == 'earliest':
            _check(_bf.bfRingSequenceOpenEarliest(self.obj, ring.obj, guarantee))
        elif which == 'at':
            _check(_bf.bfRingSequenceOpenAt(self.obj, ring.obj, time_tag, guarantee))
        #elif which == 'next':
        #    self._check( self.lib.bfRingSequenceOpenNext(pointer(self.obj), other_obj) )
        else:
            raise ValueError("Invalid 'which' parameter; must be one of: 'specific', 'latest', 'earliest'")
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.close()
    def close(self) -> None:
        _check(_bf.bfRingSequenceClose(self.obj))
    #def __next__(self):
    #    return self.next()
    #def next(self):
    #    return ReadSequence(self._ring, which='next', other_obj=self.obj)
    def increment(self) -> None:
        #self._check( self.lib.bfRingSequenceNext(pointer(self.obj)) )
        _check(_bf.bfRingSequenceNext(self.obj))
    def acquire(self, offset: int, size: int) -> "ReadSpan":
        return ReadSpan(self, offset, size)
    def read(self, span_size: int, stride: Optional[int]=None, begin: int=0) -> "ReadSpan":
        if stride is None:
            stride = span_size
        offset = begin
        while True:
            try:
                with self.acquire(offset, span_size) as ispan:
                    yield ispan
                offset += stride
            except EndOfDataStop:
                return

class SpanBase(object):
    def __init__(self, ring: Ring, writeable: bool):
        self._ring = ring
        self.writeable = writeable
    @property
    def _base_obj(self):
        return ctypes.cast(self.obj, _bf.BFspan)
    @property
    def ring(self) -> Ring:
        return self._ring
    @property
    def size(self) -> int:
        return _get(_bf.bfRingSpanGetSize, self._base_obj)
    @property
    def stride(self) -> int:
        return _get(_bf.bfRingSpanGetStride, self._base_obj)
    @property
    def offset(self) -> int:
        return _get(_bf.bfRingSpanGetOffset, self._base_obj)
    @property
    def nringlet(self) -> int:
        return _get(_bf.bfRingSpanGetNRinglet, self._base_obj)
    @property
    def _data_ptr(self):
        return _get(_bf.bfRingSpanGetData, self._base_obj)
    @property
    def data(self) -> ndarray:
        return self.data_view()
    def data_view(self, dtype: Union[str,np.dtype]=np.uint8,
                  shape: Union[int,List[int],Tuple[int]]=-1) -> ndarray:
        itemsize = DataType(dtype).itemsize
        assert( self.size   % itemsize == 0 )
        assert( self.stride % itemsize == 0 )
        data_ptr = self._data_ptr
        span_size  = self.size
        nringlet   = self.nringlet
        # TODO: We should really map the actual ring memory space and index
        #         it with offset rather than mapping from the current pointer.
        _shape   = (nringlet, span_size // itemsize)
        strides = (self.stride, itemsize) if nringlet > 1 else None
        space   = self.ring.space
        
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
    #    return self._sequence

class WriteSpan(SpanBase):
    def __init__(self,
                 ring: Ring,
                 size: int,
                 nonblocking: bool=False):
        SpanBase.__init__(self, ring, writeable=True)
        self.obj = _bf.BFwspan()
        _check(_bf.bfRingSpanReserve(self.obj, ring.obj, size, nonblocking))
        self.commit_size = size
    def commit(self, size: int) -> None:
        self.commit_size = size
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.close()
    def close(self) -> None:
        _check(_bf.bfRingSpanCommit(self.obj, self.commit_size))

class ReadSpan(SpanBase):
    def __init__(self, sequence: ReadSequence, offset: int, size: int):
        SpanBase.__init__(self, sequence.ring, writeable=False)
        self.obj = _bf.BFrspan()
        _check(_bf.bfRingSpanAcquire(self.obj, sequence.obj, offset, size))
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.release()
    def release(self) -> None:
        _check(_bf.bfRingSpanRelease(self.obj))
