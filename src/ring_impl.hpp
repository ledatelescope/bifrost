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

#pragma once

#include <bifrost/ring.h>
#include "assert.hpp"
#include "proclog.hpp"

#include <stdexcept>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <string>
#include <map>
#include <queue>
#include <set>
#include <memory>

#ifndef BF_NUMA_ENABLED
#define BF_NUMA_ENABLED 0
#endif

class BFsequence_impl;
class BFspan_impl;
class BFrspan_impl;
class BFwspan_impl;
class RingReallocLock;
class Guarantee;
typedef std::shared_ptr<BFsequence_impl> BFsequence_sptr;

class BFring_impl {
	friend class BFrsequence_impl;
	friend class BFwsequence_impl;
	friend class RingReallocLock;
	friend class Guarantee;
	
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
	
	BFoffset       _ghost_dirty_beg;
	
	bool     _writing_begun;
	bool     _writing_ended;
	BFoffset _eod;
	
	typedef std::mutex                   mutex_type;
	typedef std::lock_guard<mutex_type>  lock_guard_type;
	typedef std::unique_lock<mutex_type> unique_lock_type;
	typedef std::condition_variable      condition_type;
	typedef RingReallocLock              realloc_lock_type;
	mutable mutex_type     _mutex;
	condition_type _read_condition;
	condition_type _write_condition;
	condition_type _write_close_condition;
	condition_type _realloc_condition;
	mutable condition_type _sequence_condition;
	
	BFsize         _nread_open;
	BFsize         _nwrite_open;
	BFsize         _nrealloc_pending;

	int            _core;    	
	ProcLog        _size_log;
	
	std::queue<BFsequence_sptr>           _sequence_queue;
	std::map<std::string,BFsequence_sptr> _sequence_map;
	std::map<BFoffset,BFsequence_sptr>    _sequence_time_tag_map;
	
	typedef std::map<BFoffset,BFsize> guarantee_set; // offset-->count
	guarantee_set _guarantees;
	
	//BFoffset _wrap_offset(BFoffset offset) const;
	BFoffset _buf_offset( BFoffset offset) const;
	pointer  _buf_pointer(BFoffset offset) const;
	void _ghost_write(BFoffset offset, BFsize size);
	void _ghost_read( BFoffset offset, BFsize size);
	void _copy_to_ghost(  BFoffset buf_offset, BFsize span);
	void _copy_from_ghost(BFoffset buf_offset, BFsize span);
	bool _advance_reserve_head(unique_lock_type& lock, BFsize size, bool nonblocking);
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
	
	bool _sequence_still_within_ring(BFsequence_sptr sequence) const;
	BFoffset _get_start_of_sequence_within_ring(BFsequence_sptr sequence) const;
	BFsequence_sptr _get_earliest_or_latest_sequence(unique_lock_type& lock, bool latest) const;
	BFsequence_sptr open_earliest_or_latest_sequence(bool with_guarantee,
	                                                 std::unique_ptr<Guarantee>& guarantee,
	                                                 bool latest);
	BFsequence_sptr _get_next_sequence(BFsequence_sptr sequence,
	                                   unique_lock_type& lock) const;
	void increment_sequence_to_next(BFsequence_sptr& sequence,
	                                std::unique_ptr<Guarantee>& guarantee);
	BFsequence_sptr _get_sequence_by_name(const char* name);
	BFsequence_sptr open_sequence_by_name(const char* name,
	                                      bool with_guarantee,
	                                      std::unique_ptr<Guarantee>& guarantee);
	BFsequence_sptr _get_sequence_at(BFoffset time_tag);
	BFsequence_sptr open_sequence_at(BFoffset time_tag,
	                                 bool with_guarantee,
	                                 std::unique_ptr<Guarantee>& guarantee);
	void finish_sequence(BFsequence_sptr sequence,
	                     BFoffset offset_from_head);
	
	// No copy or move
	BFring_impl(BFring_impl const& )            = delete;
	BFring_impl& operator=(BFring_impl const& ) = delete;
	BFring_impl(BFring_impl&& )                 = delete;
	BFring_impl& operator=(BFring_impl&& )      = delete;
	
	void _write_proclog_entry();
public:
	BFring_impl(const char* name,
	            BFspace space);
	~BFring_impl();
	void resize(BFsize max_contiguous_span,
	            BFsize max_total_size,
	            BFsize max_ringlets);
	inline const char* name() const { return _name.c_str(); }
	inline BFspace space()    const { return _space; }
	inline void set_core(int core)  { _core = core; }
	inline int      core()    const { return _core; }
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
	
	inline BFoffset current_tail_offset() const {
		lock_guard_type lock(_mutex);
		return _tail;
	}
	inline BFsize current_stride() const {
		lock_guard_type lock(_mutex);
		return _stride;
	}
	inline BFsize current_nringlet() const {
		lock_guard_type lock(_mutex);
		return _nringlet;
	}
	
	BFsequence_sptr begin_sequence(const char* name,
	                               BFoffset    time_tag,
	                               BFsize      header_size,
	                               const void* header,
	                               BFsize      nringlet,
	                               BFoffset    offset_from_head=0);
	
	void reserve_span(BFsize size, BFoffset* begin, void** data, bool nonblocking);
	void commit_span(BFoffset begin, BFsize reserve_size, BFsize commit_size);
	
	void acquire_span(BFrsequence sequence,
	                  BFoffset    offset,
	                  BFsize*     size,
	                  BFoffset*   begin,
	                  void**      data);
	void release_span(BFrsequence sequence,
	                  BFoffset    begin,
	                  BFsize      size);
};

// A scoped guarantee object
class Guarantee {
	BFring   _ring;
	BFoffset _offset;
	void create(BFoffset offset)  {
		_offset = offset;
		_ring->_add_guarantee(_offset);
	}
	void destroy() { _ring->_remove_guarantee(_offset); }
public:
	Guarantee(Guarantee const& ) = delete;
	Guarantee& operator=(Guarantee const& ) = delete;
	explicit Guarantee(BFring ring)
		: _ring(ring) {
		BFring_impl::lock_guard_type lock(_ring->_mutex);
		this->create(_ring->_tail);
	}
	~Guarantee() {
		BFring_impl::lock_guard_type lock(_ring->_mutex);
		this->destroy();
	}
	void move_nolock(BFoffset offset) {
		this->destroy();
		this->create(offset);
	}
	BFoffset offset() const { return _offset; }
};
inline std::unique_ptr<Guarantee> new_guarantee(BFring ring) {
	// TODO: Use std::make_unique here (requires C++14)
	return std::unique_ptr<Guarantee>(new Guarantee(ring));
}

class BFsequence_impl {
	friend class BFring_impl;
	enum { BF_SEQUENCE_OPEN = (BFoffset)-1 };
	BFring            _ring;
	std::string       _name;
	BFoffset          _time_tag;
	BFsize            _nringlet;
	BFoffset          _begin;
	BFoffset          _end;
	typedef std::vector<char> header_type;
	header_type       _header;
	BFsequence_sptr   _next;
	BFsize            _readrefcount;
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
protected:
	BFsequence_sptr _sequence;
public:
	inline BFsequence_wrapper(BFsequence_sptr sequence) : _sequence(sequence) {}
	inline BFsequence_sptr sequence() const { return _sequence; }
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
		this->ring()->finish_sequence(_sequence, _end_offset_from_head);
	}
	void set_end_offset_from_head(BFoffset end_offset_from_head) {
		_end_offset_from_head = end_offset_from_head;
	}
};

class BFrsequence_impl : public BFsequence_wrapper {
	std::unique_ptr<Guarantee> _guarantee;
public:
	// TODO: See if can make these function bodies a bit more concise
	static BFrsequence_impl earliest_or_latest(BFring ring, bool with_guarantee, bool latest) {
		std::unique_ptr<Guarantee> guarantee;
		BFsequence_sptr sequence =
			ring->open_earliest_or_latest_sequence(with_guarantee, guarantee, latest);
		return BFrsequence_impl(sequence, guarantee);
	}
	static BFrsequence_impl by_name(BFring ring, const char* name, bool with_guarantee) {
		std::unique_ptr<Guarantee> guarantee;
		BFsequence_sptr sequence =
			ring->open_sequence_by_name(name, with_guarantee, guarantee);
		return BFrsequence_impl(sequence, guarantee);
	}
	static BFrsequence_impl at(BFring ring, BFoffset time_tag, bool with_guarantee) {
		std::unique_ptr<Guarantee> guarantee;
		BFsequence_sptr sequence =
			ring->open_sequence_at(time_tag, with_guarantee, guarantee);
		return BFrsequence_impl(sequence, guarantee);
	}
	inline BFrsequence_impl(BFsequence_sptr sequence,
	                        std::unique_ptr<Guarantee>& guarantee)
		: BFsequence_wrapper(sequence), _guarantee(std::move(guarantee)) {}
	
	inline void increment_to_next() {
		_sequence->ring()->increment_sequence_to_next(_sequence, _guarantee);
	}
	inline std::unique_ptr<Guarantee>&       guarantee()       { return _guarantee; }
	inline std::unique_ptr<Guarantee> const& guarantee() const { return _guarantee; }
	/*
	  // TODO: This is needed for bfRingSequenceOpenSame, but it's not clear
	  //         that that API is really needed. Also need to delete
	  //         assignment and move constructors if this is implemented.
	// Copy constructor points to same underlying BFsequence_impl object, but
	//   creates its own guarantee.
	BFrsequence_impl(BFrsequence_impl const& other)
		: BFsequence_wrapper(other.sequence()),
		  _guarantee(new Guarantee(*other._guarantee)) {}
	*/
};

class BFspan_impl {
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
	BFspan_impl(BFring ring,
	            BFsize size)
		: _ring(ring),
		  _size(size) {}
	virtual ~BFspan_impl() {}
	inline BFring     ring()     const { return _ring; }
	inline BFsize     size()     const { return _size; }
	// Note: These two are only safe to read while a span is open (preventing resize)
	inline BFsize     stride()   const { return _ring->current_stride(); }
	inline BFsize     nringlet() const { return _ring->current_nringlet(); }
	virtual void*     data()     const = 0;
	virtual BFoffset  offset()   const = 0;
};
class BFwspan_impl : public BFspan_impl {
	BFoffset        _begin;
	BFsize          _commit_size;
	void*           _data;
	// No copy or move
	BFwspan_impl(BFwspan_impl const& )            = delete;
	BFwspan_impl& operator=(BFwspan_impl const& ) = delete;
	BFwspan_impl(BFwspan_impl&& )                 = delete;
	BFwspan_impl& operator=(BFwspan_impl&& )      = delete;
public:
	BFwspan_impl(BFring ring,
	             BFsize size,
	             bool   nonblocking);
	~BFwspan_impl();
	BFwspan_impl* commit(BFsize size);
	inline virtual void*           data()     const { return _data; }
	// Note: This is the offset relative to the beginning of the ring,
	//         as wspans aren't firmly associated with a sequence.
	// TODO: This is likely to be confusing compared to BFrspan_impl::offset
	//         Can't easily change the name though because it's a shared API
	inline virtual BFoffset        offset()   const { return _begin; }
};
class BFrspan_impl : public BFspan_impl {
	BFrsequence     _sequence;
	BFoffset        _begin;
	void*           _data;
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
	inline BFsize size_overwritten() const {
		if( _sequence->guarantee() ) {
			return 0;
		}
		BFoffset tail = this->ring()->current_tail_offset();
		return std::max(std::min(BFdelta(tail - _begin),
		                         BFdelta(this->size())),
		                BFdelta(0));
	}
	inline virtual void*    data()     const { return _data; }
	// Note: This is the offset relative to the beginning of the sequence
	inline virtual BFoffset offset()   const { return _begin - _sequence->begin(); }
};
