/*
 * Copyright (c) 2016, The Bifrost Authors. All rights reserved.
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

// TODO: This code is a bit of a mess in some areas; consider tidying/refactoring
//         E.g., the abundant use of friend is probably a bad thing

// WARNING: Must recompile ring.cpp after modifying this file due to inlines!

#pragma once

#include <bifrost/ring.h>
#include "assert.hpp"

#include <stdexcept>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <string>
#include <map>
#include <queue>
#include <set>
#include <memory>

class BFsequence_impl;
class BFspan_impl;
class BFrspan_impl;
class BFwspan_impl;
class RingReallocLock;
typedef std::shared_ptr<BFsequence_impl> BFsequence_sptr;
/*
struct BFsequence_sptr : public std::shared_ptr<BFsequence_impl> {
private:
	typedef std::shared_ptr<BFsequence_impl> super_type;
public:
	BFsequence_sptr() : super_type() {}
	template<typename Y>
	BFsequence_sptr(Y* ptr) : super_type(ptr) {}
};
*/
class BFring_impl {
	friend class BFsequence_impl;
	friend class BFrsequence_impl;
	friend class BFspan_impl;
	friend class BFrspan_impl;
	friend class BFwspan_impl;
	friend class RingReallocLock;
	
	std::string    _name;
	BFspace        _space;
	
	typedef uint8_t*             pointer;
	typedef uint8_t const* const_pointer;
	pointer        _buf;
	
	BFsize         _ghost_span;
	BFsize         _span;
	BFsize         _stride;
	BFsize         _nringlet;
	BFoffset       _offset0;
	
	BFoffset       _tail;
	BFoffset       _head;
	BFoffset       _reserve_head;
	
	bool           _ghost_dirty;
	
	bool     _writing_begun;
	bool     _writing_ended;
	BFoffset _eod;
	
	typedef std::mutex                   mutex_type;
	typedef std::lock_guard<std::mutex>  lock_guard_type;
	typedef std::unique_lock<std::mutex> unique_lock_type;
	typedef std::condition_variable      condition_type;
	typedef RingReallocLock              realloc_lock_type;
	mutex_type     _mutex;
	condition_type _read_condition;
	condition_type _write_condition;
	condition_type _write_close_condition;
	condition_type _realloc_condition;
	condition_type _sequence_condition;
	
	BFsize         _nread_open;
	BFsize         _nwrite_open;
	BFsize         _nrealloc_pending;
	
	std::queue<BFsequence_sptr>           _sequence_queue;
	std::map<std::string,BFsequence_sptr> _sequence_map;
	std::map<BFoffset,BFsequence_sptr>    _sequence_time_tag_map;
	//typedef std::pair<BFoffset,BFsize>          guarantee_value_type;
	//typedef BFoffset guarantee_value_type;
	//typedef std::multiset<guarantee_value_type> guarantee_set;
	typedef std::map<BFoffset,BFsize> guarantee_set; // offset-->count
	guarantee_set _guarantees;
	
	BFoffset _wrap_offset(BFoffset offset) const;
	//BFoffset _advance_offset(BFoffset offset, BFdelta amount) const;
	BFoffset _buf_offset( BFoffset offset) const;
	pointer  _buf_pointer(BFoffset offset) const;
	void _ghost_write(BFoffset offset, BFsize size);
	void _ghost_read( BFoffset offset, BFsize size);
	void _copy_to_ghost(  BFoffset buf_offset, BFsize span);
	void _copy_from_ghost(BFoffset buf_offset, BFsize span);
	void _pull_tail(unique_lock_type& lock);
	inline void _add_guarantee(BFoffset offset) {
		auto iter = _guarantees.find(offset);
		if( iter == _guarantees.end() ) {
			_guarantees.insert(std::make_pair(offset, 1));
		}
		else {
			++iter->second;
		}
	}
	inline void _remove_guarantee(BFoffset offset) {
		auto iter = _guarantees.find(offset);
		if( iter == _guarantees.end() ) {
			throw BFexception(BF_STATUS_INTERNAL_ERROR);
		}
		if( !--iter->second ) {
			_guarantees.erase(iter);
			_write_condition.notify_all();
		}
	}
	inline BFoffset _get_earliest_guarantee() {
		return _guarantees.begin()->first;
	}
	void open_sequence(BFsequence_sptr sequence,
	                   BFbool          guarantee,
	                   BFoffset*       guarantee_begin);
	void close_sequence(BFsequence_sptr sequence,
	                    BFbool          guarantee,
	                    BFoffset        guarantee_begin);
	// No copy or move
	BFring_impl(BFring_impl const& )            = delete;
	BFring_impl& operator=(BFring_impl const& ) = delete;
	BFring_impl(BFring_impl&& )                 = delete;
	BFring_impl& operator=(BFring_impl&& )      = delete;
public:
	BFring_impl(const char* name,
	            BFspace space);
	~BFring_impl();
	void resize(BFsize max_contiguous_span,
	            BFsize max_total_size,
	            BFsize max_ringlets);
	inline const char* name() const { return _name.c_str(); }
	inline BFspace space()    const { return _space; }
	//inline BFsize nringlet() const { return _nringlet; }
	inline void   lock()   { _mutex.lock(); }
	inline void   unlock() { _mutex.unlock(); }
	inline void*  locked_data()            const { return _buf; }
	inline BFsize locked_contiguous_span() const { return _ghost_span; }
	inline BFsize locked_total_span()      const { return _span; }
	inline BFsize locked_nringlet()        const { return _nringlet; }
	inline BFsize locked_stride()          const { return _stride; }
	// TODO: Add getters for debugging/monitoring queries
	//         such as positions of tail, head etc. in buffer.
	
	void begin_writing();
	void end_writing();
	inline bool writing_ended() { return _writing_ended; }
	
	BFsequence_sptr begin_sequence(const char* name,
	                               BFoffset    time_tag,
	                               BFsize      header_size,
	                               const void* header,
	                               BFsize      nringlet,
	                               BFoffset    offset_from_head=0);
	BFsequence_sptr get_sequence(const char* name);
	BFsequence_sptr get_sequence_at(BFoffset time_tag);
	BFsequence_sptr get_latest_sequence();
	BFsequence_sptr get_earliest_sequence();
	
	void reserve_span(BFsize size, BFoffset* begin, void** data);
	void commit_span(BFoffset begin, BFsize reserve_size, BFsize commit_size);
	
	//void acquire_span(BFoffset offset, BFsize* size, BFbool guarantee);
	void acquire_span(BFrsequence sequence,
	                  BFoffset    offset,
	                  BFsize*     size,
	                  BFoffset*   begin,
	                  void**      data);
	void release_span(BFrsequence sequence,
	                  BFoffset    begin,
	                  BFsize      size);
};

/*
  TODO: Sequence reference/lifetime management
          typedef std::shared_ptr<BFsequence_impl>* BFsequence;
    SequenceBegin:
      *sequence = new BFsequence_impl();
      ring->
    SequenceEnd:
      mark sequence as finished
    SequenceOpen:
      fail if sequence not found (possibly due to it falling off the tail)
      seq.get_ref();
    SequenceClose:
      seq.release_ref();
    sequence falls off tail:
      remove seq refs from ring
      seq.release_ref();
 */

/*
wseq:
  begin
  end
rseq:
  open
  close
  next
*/

class BFsequence_impl {
	friend class BFring_impl;
	friend class BFspan_impl;
	friend class BFrspan_impl;
	friend class BFwspan_impl;
	enum { BF_SEQUENCE_OPEN = (BFoffset)-1 };
	BFring            _ring;
	std::string       _name;
	BFoffset          _time_tag;
	BFsize            _nringlet;
	BFoffset          _begin;
	BFoffset          _end;
	typedef std::vector<char> header_type;
	header_type _header;
	//std::shared_ptr<header_type> _header;
	//BFsequence_sptr   _next;
	BFsequence_sptr   _next;
	BFsize            _readrefcount;
	// No copy or move
	//BFsequence_impl(BFsequence_impl const& )            = delete;
	//BFsequence_impl& operator=(BFsequence_impl const& ) = delete;
	//BFsequence_impl(BFsequence_impl&& )                 = delete;
	//BFsequence_impl& operator=(BFsequence_impl&& )      = delete;
public:
	BFsequence_impl(BFring      ring,
	                const char* name,
	                BFoffset    time_tag,
	                BFsize      header_size,
	                const void* header,
	                BFsize      nringlet,
	                BFoffset    begin);
	void               finish(BFoffset offset_from_head=0);
	void               close();
	BFsequence_sptr    get_next() const;
	void               set_next(BFsequence_sptr next);
	inline bool        is_finished() const { return _end != BF_SEQUENCE_OPEN; }
	inline BFring      ring()              { return _ring; }
	inline const char* name()        const { return _name.c_str(); }
	inline BFoffset    time_tag()    const { return _time_tag; }
	inline const void* header()      const { return _header.size() ? &_header[0] : nullptr; }
	inline BFsize      header_size() const { return _header.size(); }
	inline BFsize      nringlet()    const { return _nringlet; }
	inline BFoffset    begin()       const { return _begin; }
	inline BFoffset    end()         const { return _end; }
};

class BFsequence_wrapper {
	friend class BFwspan_impl;
	friend class BFrspan_impl;
	friend class BFring_impl;
	//protected:
	// TODO: Should be shared_ptr<BFsequence_impl>?
	//typedef BFsequence pointer;
private:
	BFsequence_sptr _sequence;
	// TODO: This class is actually copy-assignable
	//BFsequence_wrapper(BFsequence_wrapper const& )            = delete;
	//BFsequence_wrapper& operator=(BFsequence_wrapper const& ) = delete;
	//BFsequence_wrapper(BFsequence_wrapper&& )                 = delete;
	//BFsequence_wrapper& operator=(BFsequence_wrapper&& )      = delete;
protected:
	BFsequence_sptr sequence() const { return _sequence; }
	inline void reset_sequence(BFsequence_sptr sequence) {
		_sequence = sequence;
	}
public:
	inline BFsequence_wrapper(BFsequence_sptr sequence) : _sequence(sequence) {}
	inline bool        is_finished() const { return _sequence->is_finished(); }
	inline BFring      ring()              { return _sequence->ring(); }
	inline const char* name()        const { return _sequence->name(); }
	inline BFoffset    time_tag()    const { return _sequence->time_tag(); }
	inline const void* header()      const { return _sequence->header(); }
	inline BFsize      header_size() const { return _sequence->header_size(); }
	inline BFsize      nringlet()    const { return _sequence->nringlet(); }
	inline BFoffset    begin()       const { return _sequence->begin(); }
};
class BFwsequence_impl : public BFsequence_wrapper {
	BFoffset _end_offset_from_head;
	BFwsequence_impl(BFwsequence_impl const& )            = delete;
	BFwsequence_impl& operator=(BFwsequence_impl const& ) = delete;
	BFwsequence_impl(BFwsequence_impl&& )                 = delete;
	BFwsequence_impl& operator=(BFwsequence_impl&& )      = delete;
public:
	inline BFwsequence_impl(BFring      ring,
	                        const char* name,
	                        BFoffset    time_tag,
	                        BFsize      header_size,
	                        const void* header,
	                        BFsize      nringlet,
	                        BFoffset    offset_from_head=0)
		: BFsequence_wrapper(ring->begin_sequence(name, time_tag,
		                                          header_size,
		                                          header, nringlet,
		                                          offset_from_head)),
		  _end_offset_from_head(0) {}
	~BFwsequence_impl() {
		this->sequence()->finish(_end_offset_from_head);
	}
	void set_end_offset_from_head(BFoffset end_offset_from_head) {
		_end_offset_from_head = end_offset_from_head;
	}
};
class BFrsequence_impl : public BFsequence_wrapper {
	friend class BFring_impl;
	BFbool   _guaranteed;
	BFoffset _guarantee_begin;
	BFbool   _is_open;
	void set_guarantee_begin(BFoffset b) { _guarantee_begin = b; }
	//BFrsequence_impl(BFrsequence_impl const& )            = delete;
	BFrsequence_impl& operator=(BFrsequence_impl const& ) = delete;
	BFrsequence_impl(BFrsequence_impl&& )                 = delete;
	BFrsequence_impl& operator=(BFrsequence_impl&& )      = delete;
	inline void open() {
		if( _is_open ) {
			throw BFexception(BF_STATUS_INTERNAL_ERROR);
		}
		_is_open = true;
		this->sequence()->ring()->open_sequence(this->sequence(),
		                                        _guaranteed,
		                                        &_guarantee_begin);
	}
	inline void close() {
		if( !_is_open ) {
			throw BFexception(BF_STATUS_INTERNAL_ERROR);
		}
		_is_open = false;
		this->sequence()->ring()->close_sequence(this->sequence(), _guaranteed, _guarantee_begin);
	}
	inline BFsequence_sptr get_next() {
		// Blocks until _next is set
		return this->sequence()->get_next();
	}
public:
	inline BFrsequence_impl(BFsequence_sptr sequence, BFbool guarantee)
		: BFsequence_wrapper(sequence), _guaranteed(guarantee), _is_open(false) {
		//this->sequence()->ring()->open_sequence(sequence,
		//                                      _guaranteed, &_guarantee_begin);
		this->open();
		/*
		// ***TODO: Replace this with inserting the guarantee inside open_*_sequence
		//            The guarantee must go at max(sequence_begin, ring->_tail)
		if( _guaranteed ) {
			//guarantee_iter =
			// TODO: Is it actually important to include size in guarantee keys?
			BFsize size=0;
			this->sequence()->ring()->_guarantees.insert(std::make_pair(sequence->begin(), size));
		}
		*/
	}
	// Copy constructor points to same underlying BFsequence_impl object, but
	//   creates its own guarantee.
	inline BFrsequence_impl(BFrsequence_impl const& other)
		: BFsequence_wrapper(other.sequence()),
		  _guaranteed(other._guaranteed),
		  _guarantee_begin(other._guarantee_begin), _is_open(false) {
		//this->sequence()->ring()->open_sequence(this->sequence(),
		//                                        _guaranteed, &_guarantee_begin);
		this->open();
		//if( _guaranteed ) {
		//	BFsize size=0;
		//	this->sequence()->ring()->_guarantees.insert(std::make_pair(this->sequence()->begin(), size));
		//}
	}
	inline ~BFrsequence_impl() {
		if( _is_open ) {
			this->close();
		}
		//if( _guaranteed ) {
		//	BFsize size = 0;
		//	this->sequence()->ring()->_guarantees.erase(this->sequence()->ring()->_guarantees.find(std::make_pair(this->sequence()->begin(), size)));
		//}
	}
	inline void increment_to_next() {
		// TODO: Is it possible/necessary for this to be atomic?
		//         Only relevant when no rspans are opened (which is a pathological case)?
		this->close();
		this->reset_sequence(this->get_next());
		this->open();
	}
	inline BFbool   guaranteed()      const { return _guaranteed; }
	inline BFoffset guarantee_begin() const { return _guarantee_begin; }
};

class BFspan_impl {
	//BFsequence _sequence;
	//BFsequence_sptr _sequence;
	BFring     _ring;
	BFsize     _size;
	// No copy or move
	BFspan_impl(BFspan_impl const& )            = delete;
	BFspan_impl& operator=(BFspan_impl const& ) = delete;
	BFspan_impl(BFspan_impl&& )                 = delete;
	BFspan_impl& operator=(BFspan_impl&& )      = delete;
protected:
	// WAR for awkwardness in subclass constructors
	void set_base_size(BFsize size) { _size = size; }
public:
	BFspan_impl(//BFsequence_sptr sequence,
	            BFring ring,
	            BFsize size)
	//BFspan_impl(BFring ring, BFsize size)
		: //_sequence(sequence),
		  //_ring(sequence->ring()),
		  _ring(ring),
		  _size(size) {}
	virtual ~BFspan_impl() {}
	//inline BFsequence sequence() const { return _sequence; }
	inline BFring     ring()     const { return _ring; }
	inline BFsize     size()     const { return _size; }
	// Note: This is only safe to read while a span is open (preventing resize)
	inline BFsize     stride()   const {
		BFring_impl::lock_guard_type lock(_ring->_mutex);
		return _ring->_stride;
	}
	inline BFsize     nringlet() const {
		BFring_impl::lock_guard_type lock(_ring->_mutex);
		return _ring->_nringlet;
	}
	//inline BFsequence_sptr sequence() const { return _sequence; }
	//virtual BFsequence_sptr sequence() const = 0;
	virtual void*           data()     const = 0;
	virtual BFoffset        offset()   const = 0;
};
class BFwspan_impl : public BFspan_impl {
	//BFsequence_sptr _sequence;
	//BFwsequence     _sequence;
	BFoffset        _begin;
	BFsize          _commit_size;
	void*           _data;
	// No copy or move
	BFwspan_impl(BFwspan_impl const& )            = delete;
	BFwspan_impl& operator=(BFwspan_impl const& ) = delete;
	BFwspan_impl(BFwspan_impl&& )                 = delete;
	BFwspan_impl& operator=(BFwspan_impl&& )      = delete;
public:
	BFwspan_impl(//BFwsequence sequence,
	             BFring      ring,
	             BFsize      size);
	~BFwspan_impl();
	BFwspan_impl* commit(BFsize size);
	//inline virtual BFsequence_sptr sequence() const { return _sequence; }
	inline virtual void*           data()     const { return _data; }
	// Note: This is the offset relative to the beginning of the ring,
	//         as wspans aren't firmly associated with a sequence.
	// TODO: This is likely to be confusing compared to BFrspan_impl::offset
	inline virtual BFoffset        offset()   const { return _begin; }
};
class BFrspan_impl : public BFspan_impl {
	//BFsequence_sptr _sequence;
	BFrsequence     _sequence;
	BFoffset        _begin;
	void*           _data;
	//BFbool          _guaranteed;
	//void _open_at(BFoffset offset, BFsize size, BFbool guarantee,
	//              BFring_impl::unique_lock_type& lock);
	////void _open();
	//void _close();
	// No copy or move
	BFrspan_impl(BFrspan_impl const& )            = delete;
	BFrspan_impl& operator=(BFrspan_impl const& ) = delete;
	BFrspan_impl(BFrspan_impl&& )                 = delete;
	BFrspan_impl& operator=(BFrspan_impl&& )      = delete;
public:
	BFrspan_impl(BFrsequence sequence,
	             BFoffset    offset,
	             BFsize      size);
	~BFrspan_impl();
	//void advance(BFdelta delta, BFsize size, BFbool guarantee);
	//inline virtual BFsequence_sptr sequence() const { return _sequence; }
	inline virtual void*           data()     const { return _data; }
	// Note: This is the offset relative to the beginning of the sequence
	inline virtual BFoffset        offset()   const { return _begin - _sequence->begin(); }
};
