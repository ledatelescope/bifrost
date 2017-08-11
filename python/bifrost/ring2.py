
# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
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

# TODO: Some of this code has gotten a bit hacky
#         Also consider merging some of the logic into the backend

from libbifrost import _bf, _check, _get, BifrostObject, _string2space, _space2string
from DataType import DataType
from ndarray import ndarray, _address_as_buffer
from copy import copy, deepcopy

import ctypes
import numpy as np

try:
    import simplejson as json
except ImportError:
    print "WARNING: Install simplejson for better performance"
    import json

# TODO: Should probably move this elsewhere (e.g., utils)
def split_shape(shape):
    """Splits a shape into its ringlet shape and frame shape
    E.g., (2,3,-1,4,5) -> (2,3), (4,5)
    """
    ringlet_shape = []
    for i, dim in enumerate(shape):
        if dim == -1:
            frame_shape = shape[i + 1:]
            return ringlet_shape, frame_shape
        ringlet_shape.append(dim)
    raise ValueError("No time dimension (-1) found in shape")

def compose_unary_funcs(f, g):
    return lambda x: f(g(x))

def ring_view(ring, header_transform):
    new_ring = ring.view()
    old_header_transform = ring.header_transform
    if old_header_transform is not None:
        header_transform = compose_unary_funcs(header_transform,
                                               old_header_transform)
    new_ring.header_transform = header_transform
    return new_ring

class Ring(BifrostObject):
    instance_count = 0
    def __init__(self, space='system', name=None, owner=None):
        # If this is non-None, then the object is wrapping a base Ring instance
        self.base = None
        self.space = space
        if name is None:
            name = 'ring_%i' % Ring.instance_count
            Ring.instance_count += 1
        BifrostObject.__init__(self, _bf.bfRingCreate, _bf.bfRingDestroy,
                               name, _string2space(self.space))
        self.owner = owner
        self.header_transform = None
    def __del__(self):
        if self.base is not None:
            BifrostObject.__del__(self)
    def view(self):
        new_ring = copy(self)
        new_ring.base = self
        return new_ring
    def resize(self, contiguous_bytes, total_bytes=None, nringlet=1):
        _check( _bf.bfRingResize(self.obj,
                                 contiguous_bytes,
                                 total_bytes,
                                 nringlet) )
    @property
    def name(self):
        return _get(_bf.bfRingGetName, self.obj)
    def begin_writing(self):
        return RingWriter(self)
    def _begin_writing(self):
        _check( _bf.bfRingBeginWriting(self.obj) )
    def end_writing(self):
        _check( _bf.bfRingEndWriting(self.obj) )
    def open_sequence(self, name, guarantee=True):
        return ReadSequence(self, name=name, guarantee=guarantee)
    def open_latest_sequence(self, guarantee=True):
        return ReadSequence(self, which='latest', guarantee=guarantee)
    def open_earliest_sequence(self, guarantee=True):
        return ReadSequence(self, which='earliest', guarantee=guarantee)
    # TODO: Alternative name?
    def read(self, whence='earliest', guarantee=True):
        with ReadSequence(self, which=whence, guarantee=guarantee,
                          header_transform=self.header_transform) as cur_seq:
            while True:
                yield cur_seq
                cur_seq.increment()

class RingWriter(object):
    def __init__(self, ring):
        self.ring = ring
        self.ring._begin_writing()
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.ring.end_writing()
    def begin_sequence(self, header, buf_nframe):
        return WriteSequence(ring=self.ring,
                             header=header,
                             buf_nframe=buf_nframe)

class SequenceBase(object):
    """Python object for a ring's sequence (data unit)"""
    def __init__(self, ring):
        self._ring = ring
        self._header = None
        self._tensor = None
    @property
    def _base_obj(self):
        return ctypes.cast(self.obj, _bf.BFsequence)
    @property
    def ring(self):
        return self._ring
    @property
    def name(self):
        return _get(_bf.bfRingSequenceGetName, self._base_obj)
    @property
    def time_tag(self):
        return _get(_bf.bfRingSequenceGetTimeTag, self._base_obj)
    @property
    def nringlet(self):
        return _get(_bf.bfRingSequenceGetNRinglet, self._base_obj)
    @property
    def header_size(self):
        return _get(_bf.bfRingSequenceGetHeaderSize, self._base_obj)
    @property
    def _header_ptr(self):
        return _get(_bf.bfRingSequenceGetHeader, self._base_obj)
    @property
    def tensor(self): # TODO: This shouldn't be public
        if self._tensor is not None:
            return self._tensor
        header = self.header
        shape = header['_tensor']['shape']
        ringlet_shape, frame_shape = split_shape(shape)
        nringlet       = reduce(lambda x, y: x * y, ringlet_shape, 1)
        frame_nelement = reduce(lambda x, y: x * y, frame_shape,   1)
        dtype = header['_tensor']['dtype']
        nbit = DataType(dtype).itemsize_bits
        assert(nbit % 8 == 0)
        frame_nbyte = frame_nelement * nbit // 8
        self._tensor = {}
        self._tensor['dtype']         = DataType(dtype)
        self._tensor['ringlet_shape'] = ringlet_shape
        self._tensor['nringlet']      = nringlet
        self._tensor['frame_shape']   = frame_shape
        self._tensor['frame_nbyte']   = frame_nbyte
        self._tensor['dtype_nbyte']   = nbit // 8
        return self._tensor
    @property
    def header(self):
        if self._header is not None:
            return self._header
        size = self.header_size
        if size == 0:
            # WAR for hdr_buffer_ptr.contents crashing when size == 0
            hdr_array = np.empty(0, dtype=np.uint8)
            hdr_array.flags['WRITEABLE'] = False
            return json.loads(hdr_array.tostring())
        hdr_buffer = _address_as_buffer(self._header_ptr, size, readonly=True)
        hdr_array = np.frombuffer(hdr_buffer, dtype=np.uint8)
        hdr_array.flags['WRITEABLE'] = False
        self._header = json.loads(hdr_array.tostring())
        return self._header

class WriteSequence(SequenceBase):
    def __init__(self, ring, header, buf_nframe):
        SequenceBase.__init__(self, ring)
        self._header = header
        # This allows passing DataType instances instead of string types
        header['_tensor']['dtype'] = str(header['_tensor']['dtype'])
        header_str = json.dumps(header)
        header_size = len(header_str)
        gulp_nframe = header['gulp_nframe']
        tensor = self.tensor
        # **TODO: Consider moving this into bfRingSequenceBegin
        self.ring.resize(gulp_nframe * tensor['frame_nbyte'],
                          buf_nframe * tensor['frame_nbyte'],
                         tensor['nringlet'])
        offset_from_head = 0
        # TODO: How to allow time_tag to be optional? Probably need to plumb support through to backend.
        self.obj = _bf.BFwsequence()
        _check(_bf.bfRingSequenceBegin(
            self.obj,
            ring.obj,
            header['name'],
            header['time_tag'],
            header_size,
            header_str,
            tensor['nringlet'],
            offset_from_head))
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.end()
    def end(self):
        offset_from_head = 0
        _check(_bf.bfRingSequenceEnd(self.obj, offset_from_head))
    def reserve(self, nframe):
        return WriteSpan(self.ring, self, nframe)

class ReadSequence(SequenceBase):
    def __init__(self, ring, which='specific', name="",
                 other_obj=None, guarantee=True,
                 header_transform=None):
        SequenceBase.__init__(self, ring)
        self._ring = ring
        # A function for transforming the header before it's read
        self.header_transform = header_transform
        self.obj = _bf.BFrsequence()
        if which == 'specific':
            _check(_bf.bfRingSequenceOpen(self.obj, ring.obj, name, guarantee))
        elif which == 'latest':
            _check(_bf.bfRingSequenceOpenLatest(self.obj, ring.obj, guarantee))
        elif which == 'earliest':
            _check(_bf.bfRingSequenceOpenEarliest(self.obj, ring.obj, guarantee))
        else:
            raise ValueError("Invalid 'which' parameter; must be one of: 'specific', 'latest', 'earliest'")

    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.close()
    def close(self):
        _check(_bf.bfRingSequenceClose(self.obj))
    def increment(self):
        _check(_bf.bfRingSequenceNext(self.obj))
        # Must invalidate cached header and tensor because this is now
        #   a new sequence.
        self._header = None
        self._tensor = None
    def acquire(self, frame_offset, nframe):
        return ReadSpan(self, frame_offset, nframe)
    def read(self, nframe, stride=None, begin=0):
        if stride is None:
            stride = nframe
        offset = begin
        while True:
            with self.acquire(offset, nframe) as ispan:
                yield ispan
            offset += stride
    def resize(self, gulp_nframe, buf_nframe=None, buffer_factor=None):
        if buf_nframe is None:
            if buffer_factor is None:
                buffer_factor = 3
            buf_nframe = int(np.ceil(gulp_nframe * buffer_factor))
        tensor = self.tensor
        return self._ring.resize(gulp_nframe * tensor['frame_nbyte'],
                                 buf_nframe * tensor['frame_nbyte'])
    @property
    def header(self):
        hdr = super(ReadSequence, self).header
        if self.header_transform is not None:
            hdr = self.header_transform(deepcopy(hdr))
            if hdr is None:
                raise ValueError("Header transform returned None")
        return hdr

def accumulate(vals, op='+', init=None, reverse=False):
    if   op == '+':   op = lambda a, b: a + b
    elif op == '*':   op = lambda a, b: a * b
    elif op == 'min': op = lambda a, b: min(a, b)
    elif op == 'max': op = lambda a, b: max(a, b)
    results = []
    if reverse:
        vals = reversed(list(vals))
    for i, val in enumerate(vals):
        if i == 0:
            results.append(val if init is None else init)
        else:
            results.append(op(results[-1], val))
    if reverse:
        results = list(reversed(results))
    return results

class SpanBase(object):
    def __init__(self, ring, sequence, writeable):
        self._ring     = ring
        self._sequence = sequence
        self.writeable = writeable
        self._data = None
    def _set_base_obj(self, obj):
        self._base_obj = ctypes.cast(obj, _bf.BFspan)
        self._cache_info()
    def _cache_info(self):
        self._info = _bf.BFspan_info()
        _check(_bf.bfRingSpanGetInfo(self._base_obj, self._info))
    @property
    def ring(self):
        return self._ring
    @property
    def sequence(self):
        return self._sequence
    @property
    def tensor(self):
        return self._sequence.tensor
    @property
    def _size_bytes(self):
        # **TODO: Change back-end to use long instead of uint64_t
        return int(self._info.size)
    @property
    def _stride_bytes(self):
        # **TODO: Change back-end to use long instead of uint64_t
        return int(self._info.stride)
    @property
    def frame_nbyte(self):
        return self._sequence.tensor['frame_nbyte']
    @property
    def frame_offset(self):
        # **TODO: Change back-end to use long instead of uint64_t
        byte_offset = int(self._info.offset)
        assert(byte_offset % self.frame_nbyte == 0)
        return byte_offset // self.frame_nbyte
    @property
    def _nringlet(self):
        # **TODO: Change back-end to use long instead of uint64_t
        return int(self._info.nringlet)
    @property
    def _data_ptr(self):
        return self._info.data
    @property
    def nframe(self):
        size_bytes = self._size_bytes
        assert(size_bytes % self.sequence.tensor['frame_nbyte'] == 0)
        nframe  = size_bytes // self.sequence.tensor['frame_nbyte']
        return nframe
    @property
    def shape(self):
        shape = (self._sequence.tensor['ringlet_shape'] +
                 [self.nframe] +
                 self._sequence.tensor['frame_shape'])
        return shape
    @property
    def strides(self):
        tensor = self._sequence.tensor
        strides = [tensor['dtype_nbyte']]
        for dim in reversed(tensor['frame_shape']):
            strides.append(dim * strides[-1])
        if len(tensor['ringlet_shape']) > 0:
            strides.append(self._stride_bytes) # First ringlet dimension
        for dim in reversed(tensor['ringlet_shape'][1:]):
            strides.append(dim * strides[-1])
        strides = list(reversed(strides))
        return strides
    @property
    def dtype(self):
        return self._sequence.tensor['dtype']
    @property
    def data(self):

        # TODO: This function used to be super slow due to pyclibrary calls
        #         Check whether it still is!

        if self._data is not None:
            return self._data
        data_ptr = self._data_ptr

        space = self.ring.space

        # **TODO: Need to integrate support for endianness and conjugatedness
        #         Also need support in headers for units of the actual values,
        #           in addition to the axis scales.
        data_array = ndarray(space=space,
                             shape=self.shape,
                             strides=self.strides,
                             buffer=data_ptr,
                             dtype=self.dtype)
        data_array.flags['WRITEABLE'] = self.writeable

        return data_array

class WriteSpan(SpanBase):
    def __init__(self,
                 ring,
                 sequence,
                 nframe):
        SpanBase.__init__(self, ring, sequence, writeable=True)
        nbyte = nframe * self._sequence.tensor['frame_nbyte']
        self.obj = _bf.BFwspan()
        _check(_bf.bfRingSpanReserve(self.obj, ring.obj, nbyte))
        self._set_base_obj(self.obj)
        # Note: We default to 0 instead of nframe so that we don't accidentally
        #         commit bogus data if a block throws an exception.
        self.commit_nframe = 0
        # TODO: Why do exceptions here not show up properly?
        #raise ValueError("SHOW ME THE ERROR")
    def commit(self, nframe):
        self.commit_nframe = nframe
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.close()
    def close(self):
        commit_nbyte = self.commit_nframe * self._sequence.tensor['frame_nbyte']
        _check(_bf.bfRingSpanCommit(self.obj, commit_nbyte))

class ReadSpan(SpanBase):
    def __init__(self, sequence, frame_offset, nframe):
        SpanBase.__init__(self, sequence.ring, sequence, writeable=False)
        tensor = sequence.tensor
        self.obj = _bf.BFrspan()
        _check(_bf.bfRingSpanAcquire(
            self.obj,
            sequence.obj,
            frame_offset * tensor['frame_nbyte'],
            nframe * tensor['frame_nbyte']))
        self._set_base_obj(self.obj)
        self.nframe_skipped = min(self.frame_offset - frame_offset, nframe)
        self.requested_frame_offset = frame_offset
    @property
    def nframe_overwritten(self):
        nbyte_overwritten = ctypes.c_ulong()
        _check(_bf.bfRingSpanGetSizeOverwritten(self.obj, nbyte_overwritten))
        nbyte_overwritten = nbyte_overwritten.value
        assert(nbyte_overwritten  % self.frame_nbyte == 0)
        return nbyte_overwritten // self.frame_nbyte
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.release()
    def release(self):
        _check(_bf.bfRingSpanRelease(self.obj))
