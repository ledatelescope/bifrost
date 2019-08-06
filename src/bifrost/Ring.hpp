/*
 * Copyright (c) 2016, The Bifrost Authors. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of The Bifrost Authors nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <bifrost/Common.hpp>
#include <bifrost/ring.h>

#include <memory>

//using std::shared_ptr;

namespace bifrost {

namespace ring {

class RingLock;

class RingWrapper {
	friend class RingLock;
	friend class RingWriter;
	friend class WriteSequence;
	friend class ReadSequence;
	friend class WriteSpan;
protected:
	BFring _obj;
public:
	enum {
		DEFAULT_BUFFERING = 4
	};
	RingWrapper(BFring obj) : _obj(obj) {}
	BIFROST_DEFINE_GETTER(const char*, name,  bfRingGetName,  _obj)
	BIFROST_DEFINE_GETTER(BFspace,     space, bfRingGetSpace, _obj)
	inline void resize(BFsize contiguous_span,
	                   BFsize total_span=BFsize(-1),
	                   BFsize nringlet=1) {
		if( total_span == BFsize(-1) ) {
			total_span = DEFAULT_BUFFERING * contiguous_span;
		}
		check( bfRingResize(_obj, contiguous_span, total_span, nringlet) );
	}
	operator BFring() const { return _obj; }
};

class Ring : public RingWrapper {
	Ring(Ring const& );
	Ring& operator=(Ring const& );
public:
	inline Ring(const char *name="", BFspace space=BF_SPACE_SYSTEM) : RingWrapper(0) {
		check( bfRingCreate(&_obj, name, space) );
	}
	inline ~Ring() {
		if( _obj ) {
			bfRingDestroy(_obj);
		}
	}
};

class RingLock {
	Ring& _ring;
	RingLock(RingLock const&);
	RingLock& operator=(RingLock const&);
public:
	inline RingLock(Ring& ring) : _ring(ring) {
		check( bfRingLock(_ring._obj) );
	}
	inline ~RingLock() {
		check( bfRingUnlock(_ring._obj) );
	}
	BIFROST_DEFINE_GETTER(void*,  data,            bfRingLockedGetData,           _ring._obj)
	BIFROST_DEFINE_GETTER(BFsize, contiguous_span, bfRingLockedGetContiguousSpan, _ring._obj)
	BIFROST_DEFINE_GETTER(BFsize, total_span,      bfRingLockedGetTotalSpan,      _ring._obj)
	BIFROST_DEFINE_GETTER(BFsize, nringlet,        bfRingLockedGetNRinglet,       _ring._obj)
	BIFROST_DEFINE_GETTER(BFsize, stride,          bfRingLockedGetStride,         _ring._obj)
};

class RingWriter {
	friend class WriteSequence;
	RingWrapper& _ring;
	bool         _ended;
	RingWriter(RingWriter const&);
	RingWriter& operator=(RingWriter const&);
public:
	inline RingWriter(RingWrapper& ring) : _ring(ring), _ended(false) {
		check( bfRingBeginWriting(_ring._obj) );
	}
	inline ~RingWriter() {
		this->close();
	}
	inline void close() {
		if( !_ended ) {
			check( bfRingEndWriting(_ring._obj) );
			_ended = true;
		}
	}
	inline RingWrapper&       ring()       { return _ring; }
	inline RingWrapper const& ring() const { return _ring; }
};

namespace detail {

class Sequence {
	BFsequence _obj;
protected:
	inline Sequence() : _obj(0) {}
	inline Sequence(BFsequence obj) : _obj(obj) {}
	// WAR for awkwardness in subclass constructors
	inline void set_base_obj(BFsequence obj) { _obj = obj; }
public:
	BIFROST_DEFINE_GETTER(const char*, name,        bfRingSequenceGetName,       _obj)
	BIFROST_DEFINE_GETTER(const void*, header,      bfRingSequenceGetHeader,     _obj)
	BIFROST_DEFINE_GETTER(BFsize,      header_size, bfRingSequenceGetHeaderSize, _obj)
	BIFROST_DEFINE_GETTER(BFsize,      nringlet,    bfRingSequenceGetNRinglet,   _obj)
};

} // namespace detail

class WriteSequence : public detail::Sequence {
	//friend class WriteSpan;
	BFwsequence _obj;
	//RingWriter& _oring;
	BFoffset    _end_offset_from_head;
	WriteSequence(WriteSequence const&);
	WriteSequence& operator=(WriteSequence const&);
public:
	inline WriteSequence(RingWriter& oring,
	                     std::string name="",
	                     BFoffset    time_tag=-1,
	                     BFsize      header_size=0,
	                     void const* header=0,
	                     BFsize      nringlet=1,
	                     BFoffset    offset_from_head=0)
		: _obj(0), _end_offset_from_head(0) {//, _oring(oring) {
		check( bfRingSequenceBegin(&_obj, oring.ring()._obj, name.c_str(),
		                           time_tag,
		                           header_size, header, nringlet,
		                           offset_from_head) );
		this->set_base_obj((BFsequence)_obj);
		
	}
	inline ~WriteSequence() {
		if( _obj ) {
			check( bfRingSequenceEnd(_obj, _end_offset_from_head) );
		}
	}
	void set_end_offset(BFoffset end_offset_from_head) {
		_end_offset_from_head = end_offset_from_head;
	}
};

class ReadSequence : public detail::Sequence {
	friend class ReadSpan;
	BFrsequence _obj;
	ReadSequence(BFrsequence obj)
		: detail::Sequence((BFsequence)obj), _obj(obj) {}
	void close() {
		if( _obj ) {
			check( bfRingSequenceClose(_obj) );
			_obj = 0;
		}
	}
public:
	inline static ReadSequence open(RingWrapper const& ring, std::string name, BFbool guarantee=true) {
		BFrsequence obj; check( bfRingSequenceOpen(&obj, ring._obj, name.c_str(), guarantee) ); return ReadSequence(obj);
	}
	inline static ReadSequence open_latest(RingWrapper const& ring, BFbool guarantee=true) {
		BFrsequence obj; check( bfRingSequenceOpenLatest(&obj, ring._obj, guarantee) ); return ReadSequence(obj);
	}
	inline static ReadSequence open_earliest(RingWrapper const& ring, BFbool guarantee=true) {
		BFrsequence obj; check( bfRingSequenceOpenEarliest(&obj, ring._obj, guarantee) ); return ReadSequence(obj);
	}
	//inline static ReadSequence open_next(ReadSequence const& previous) {
	//	BFrsequence obj; check( bfRingSequenceOpenNext(&obj, previous._obj) ); return ReadSequence(obj);
	//}
	//inline static ReadSequence open_same(ReadSequence const& existing) {
	//	BFrsequence obj; check( bfRingSequenceOpenSame(&obj, existing._obj) ); return ReadSequence(obj);
	//}
	inline ReadSequence(ReadSequence&& other)
		: _obj(other._obj) {
		other._obj = 0;
	}
	inline ReadSequence(ReadSequence const& other) = delete;/*
		: _obj(0) {
		check( bfRingSequenceOpenSame(&_obj, other._obj) );
		this->set_base_obj((BFsequence)_obj);
	}*/
	inline ReadSequence& increment() {
		//check( bfRingSequenceNext(&_obj) );
		check( bfRingSequenceNext(_obj) );
		return *this;
	}
	inline ReadSequence& operator=(ReadSequence const& other) = delete;/* {
		if( &other != this ) {
			this->close();
			ReadSequence cpy(other);
			this->swap(cpy);
		}
		return *this;
	}*/
	inline void swap(ReadSequence& other) {
		std::swap(_obj, other._obj);
	}
	inline ~ReadSequence() {
		this->close();
	}
};

namespace detail {
class Span {
	BFspan _obj;
protected:
	inline Span() : _obj(0) {}
	inline Span(BFspan obj) : _obj(obj) {}
	// WAR for awkwardness in subclass constructors
	inline void set_base_obj(BFspan obj) { _obj = obj; }
public:
	BIFROST_DEFINE_GETTER(void*,      data,    bfRingSpanGetData,   _obj)
	BIFROST_DEFINE_GETTER(BFsize,     size,    bfRingSpanGetSize,   _obj)
	BIFROST_DEFINE_GETTER(BFsize,     stride,  bfRingSpanGetStride, _obj)
	BIFROST_DEFINE_GETTER(BFsize,     offset,  bfRingSpanGetOffset, _obj)
};
} // namespace detail

// Note: This defaults to committing nothing, so
//         commit() must be called explicitly.
// TODO: Could theoretically do the reverse and provide
//         an explicit cancel()=>commit(0) method. Not sure
//         which is the better approach at this stage.
//         Given that explicit commit(n) is required when
//           n<size, it may be better stick with default=cancel.
class WriteSpan : public detail::Span {
	BFwspan        _obj;
	//WriteSequence& _sequence;
	BFsize         _commit_size;
	WriteSpan(WriteSpan const&);
	WriteSpan& operator=(WriteSpan const&);
public:
	//inline WriteSpan(WriteSequence& sequence, BFsize size)
	inline WriteSpan(RingWriter& oring, BFsize size, bool nonblocking=false)
		: _obj(0),
		  //_sequence(sequence),
		  //_commit_size(size) {
		  _commit_size(0) {
		check( bfRingSpanReserve(&_obj,
		                         //sequence._obj,
		                         oring.ring()._obj,
		                         size,
		                         nonblocking) );
		this->set_base_obj((BFspan)_obj);
	}
	inline ~WriteSpan() {
		check( bfRingSpanCommit(_obj, _commit_size) );
	}
	inline void commit(BFsize size) { _commit_size = size; }
	inline void commit()            { _commit_size = this->size(); }
	//inline WriteSequence&       sequence()       { return _sequence; }
	//inline WriteSequence const& sequence() const { return _sequence; }
};

class ReadSpan : public detail::Span {
	BFrspan        _obj;
	ReadSequence&  _sequence;
	ReadSpan(ReadSpan const&);
	ReadSpan& operator=(ReadSpan const&);
public:
	inline ReadSpan(ReadSequence& sequence, BFoffset offset, BFsize size)
		: _obj(0), _sequence(sequence) {
		check( bfRingSpanAcquire(&_obj, sequence._obj, offset, size) );
		this->set_base_obj((BFspan)_obj);
	}
	inline ~ReadSpan() {
		check( bfRingSpanRelease(_obj) );
	}
	//inline ReadSequence&       sequence()       { return _sequence; }
	//inline ReadSequence const& sequence() const { return _sequence; }
};

} // namespace ring

} // namespace bifrost
