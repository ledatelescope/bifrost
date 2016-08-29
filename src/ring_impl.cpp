/*
 *  Copyright 2016 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// TODO: Confirm that resizing is completely safe in any state

// Note: Due to potential wrapping, offsets should never be compared using <,<=,>,>=
//         Always compare positive differences between offsets instead
//           E.g., offset < tail --> _head-offset > _head-tail

#include "ring_impl.hpp"
#include "utils.hpp"
#include "assert.hpp"
#include <bifrost/memory.h>

#ifdef BF_CUDA_ENABLED
	#include <cuda_runtime_api.h>
	#include <iostream>
#endif

// This implements a lock with the condition that no reads or writes
//   can be open while it is held.
class RingReallocLock {
	typedef BFring_impl::unique_lock_type unique_lock_type;
	unique_lock_type& _lock;
	BFring_impl*      _ring;
	// No copy or move
	RingReallocLock(RingReallocLock const& )            = delete;
	RingReallocLock& operator=(RingReallocLock const& ) = delete;
	RingReallocLock(RingReallocLock&& )                 = delete;
	RingReallocLock& operator=(RingReallocLock&& )      = delete;
public:
	inline RingReallocLock(unique_lock_type& lock,
	                       BFring_impl*      ring)
		: _lock(lock), _ring(ring) {
		++_ring->_nrealloc_pending;
		_ring->_realloc_condition.wait(_lock, [&]() {
			return (_ring->_nwrite_open == 0 &&
			        _ring->_nread_open == 0);
		});
	}
	inline ~RingReallocLock() {
		--_ring->_nrealloc_pending;
		_ring->_read_condition.notify_all();
		_ring->_write_condition.notify_all();
	}
};

BFring_impl::BFring_impl(BFenum space)
	: _space(space), _buf(nullptr),
	  _ghost_span(0), _span(0), _stride(0), _nringlet(0), _offset0(0),
	  _tail(0), _head(0), _reserve_head(0),
	  _ghost_dirty(false),
	  _writing_begun(false), _writing_ended(false), _eod(0),
	  _nread_open(0), _nwrite_open(0), _nrealloc_pending(0) {
	
	BF_ASSERT_EXCEPTION(space==BF_SPACE_SYSTEM       ||
#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
	          space==BF_SPACE_CUDA         ||
	          space==BF_SPACE_CUDA_HOST    ||
	          space==BF_SPACE_CUDA_MANAGED ||
#endif
	          false,
	          BF_STATUS_INVALID_ARGUMENT);
}
BFring_impl::~BFring_impl() {
	// TODO: Should check if anything is still open here?
	if( _buf ) {
		bfFree(_buf, _space);
	}
}
void BFring_impl::resize(BFsize contiguous_span,
                         BFsize total_span,
                         BFsize nringlet) {
	unique_lock_type lock(_mutex);
	// Check if reallocation is actually necessary
	if( contiguous_span <= _ghost_span &&
	    total_span      <= _span &&
	    nringlet        <= _nringlet) {
		return;
	}
	realloc_lock_type realloc_lock(lock, this);
	// Check if reallocation is still actually necessary
	if( contiguous_span <= _ghost_span &&
	    total_span      <= _span &&
	    nringlet        <= _nringlet) {
		return;
	}
	// Perform the reallocation
	BFsize  new_ghost_span = std::max(contiguous_span, _ghost_span);
	BFsize  new_span       = std::max(total_span,      _span);
	BFsize  new_nringlet   = std::max(nringlet,        _nringlet);
	//new_ghost_span = round_up(new_ghost_span, bfGetAlignment());
	//new_span       = round_up(new_span,       bfGetAlignment());
	bool span_smaller_than_alignment = (new_span < (long unsigned int)BF_ALIGNMENT);
	if (span_smaller_than_alignment)
		new_span = (long unsigned int)BF_ALIGNMENT;
	// Note: This is critical to enable safe overflowing/wrapping of offsets
	new_span = round_up_pow2(new_span);
	// This is just to ensure nice indexing
	// TODO: Not sure if this is a good idea or not
	//new_ghost_span = round_up_pow2(new_ghost_span);
	new_ghost_span = round_up(new_ghost_span, bfGetAlignment());
	BFsize  new_stride = new_span + new_ghost_span;
	BFsize  new_nbyte  = new_stride*new_nringlet;
	//pointer new_buf    = (pointer)bfMalloc(new_nbyte, _space);
	//std::cout << "new_buf = " << (void*)new_buf << std::endl; // HACK TESTING
	pointer new_buf = nullptr;
	//std::cout << "contig_span:    " << contiguous_span << std::endl;
	//std::cout << "total_span:     " << total_span << std::endl;
	//std::cout << "new_span:       " << new_span << std::endl;
	//std::cout << "new_ghost_span: " << new_ghost_span << std::endl;
	//std::cout << "new_nringlet:   " << new_nringlet << std::endl;
	//std::cout << "new_stride:     " << new_stride << std::endl;
	//std::cout << "Allocating " << new_nbyte << std::endl;
	BF_ASSERT_EXCEPTION(bfMalloc((void**)&new_buf, new_nbyte, _space) == BF_STATUS_SUCCESS,
	                    BF_STATUS_MEM_ALLOC_FAILED);
	
	if( _buf ) {
		// Must move existing data and delete old buf
		if( _buf_offset(_tail) < _buf_offset(_head) ) {
			// Copy middle to beginning
			bfMemcpy2D(new_buf,                   new_stride, _space,
			           _buf + _buf_offset(_tail),    _stride, _space,
			           BFoffset(_head - _tail), _nringlet);
			_offset0 = _tail;
		}
		else {
			// Copy beg to beg and end to end, with larger gap between
			bfMemcpy2D(new_buf, new_stride, _space,
			           _buf,       _stride, _space,
			           _buf_offset(_head), _nringlet);
			bfMemcpy2D(new_buf + (_buf_offset(_tail)+(new_span-_span)), new_stride, _space,
			           _buf    +  _buf_offset(_tail),                      _stride, _space,
			           _span - _buf_offset(_tail), _nringlet);
			_offset0 = _head - _buf_offset(_head); // TODO: Check this for sign/overflow issues
		}
		// Copy old ghost region to new buffer
		bfMemcpy2D(new_buf + new_span, new_stride, _space,
		           _buf    +    _span,    _stride, _space,
		           _ghost_span, _nringlet);
		// Copy the part of the beg corresponding to the extra ghost space
		bfMemcpy2D(new_buf + new_span + _ghost_span, new_stride, _space,
		           _buf + _ghost_span,                  _stride, _space,
		           std::min(new_ghost_span, _span) - _ghost_span, _nringlet);
		_ghost_dirty = true; // TODO: Is this the right thing to do?
		bfFree(_buf, _space);
	}
	_buf        = new_buf;
	_ghost_span = new_ghost_span;
	_span       = new_span;
	_stride     = new_stride;
	_nringlet   = new_nringlet;
}
void BFring_impl::begin_writing() {
	lock_guard_type lock(_mutex);
	BF_ASSERT_EXCEPTION(!_writing_begun && !_writing_ended, BF_STATUS_INVALID_STATE);
	_writing_begun = true;
}
void BFring_impl::end_writing() {
	lock_guard_type lock(_mutex);
	BF_ASSERT_EXCEPTION(_writing_begun && !_writing_ended, BF_STATUS_INVALID_STATE);
	BF_ASSERT_EXCEPTION(!_nwrite_open,                     BF_STATUS_INVALID_STATE);
	// TODO: Assert that no sequences are open for writing
	_writing_ended = true;
	_eod = _head;
	_sequence_condition.notify_all();
}
/*
BFoffset BFring_impl::_wrap_offset(BFoffset offset) const {
	// Avoid integer overflow by wrapping to a multiple of _span once
	//   the offset passes the half-way point of representable values.
	BFoffset halfmax = (std::numeric_limits<BFoffset>::max() - 1) / 2 + 1;
	BFoffset wrap_point = round_up(halfmax, _span);
	return (offset >= wrap_point ?
	        offset - wrap_point,
	        offset);
}
*/
//BFoffset BFring_impl::_advance_offset(BFoffset offset, BFdelta amount) const {
//	return _wrap_offset(offset + amount);
//}
BFoffset BFring_impl::_buf_offset(BFoffset offset) const {
	//while( offset < _offset0 ) {
	//	offset += _span;
	//}
	return (offset - _offset0) % _span;
}
BFring_impl::pointer BFring_impl::_buf_pointer(BFoffset offset) const {
	return _buf + _buf_offset(offset);
}
void BFring_impl::_ghost_write(BFoffset offset, BFsize span) {
	BFoffset buf_offset_beg = _buf_offset(offset);
	BFoffset buf_offset_end = _buf_offset(offset + span);
	if( buf_offset_end < buf_offset_beg ) {
		// The write went into the ghost region, so copy to the ghosted part
		this->_copy_from_ghost(0, buf_offset_end);
	}
	else if( buf_offset_beg < (BFoffset)_ghost_span ) {
		// The write touched the ghosted front of the buffer
		_ghost_dirty = true;
		// TODO: Implement fine-grained dirty region tracking
	}
}
void BFring_impl::_ghost_read(BFoffset offset, BFsize span) {
	BFoffset buf_offset_beg = _buf_offset(offset);
	BFoffset buf_offset_end = _buf_offset(offset + span);
	if( buf_offset_end < buf_offset_beg ) {
		// The read will enter the ghost region, so copy from the ghosted part
		if( _ghost_dirty ) {
			this->_copy_to_ghost(0, _ghost_span);
			_ghost_dirty = false;
		}
	}
}
void BFring_impl::_copy_to_ghost(BFoffset buf_offset, BFsize span) {
	// Copy from the front of the buffer to the ghost region at the end
	bfMemcpy2D(_buf + (_span + buf_offset), _stride, _space,
	           _buf + buf_offset,           _stride, _space,
	           span, _nringlet);
}
void BFring_impl::_copy_from_ghost(BFoffset buf_offset, BFsize span) {
	// Copy from the ghost region to the front of the buffer
	bfMemcpy2D(_buf + buf_offset,           _stride, _space,
	           _buf + (_span + buf_offset), _stride, _space,
	           span, _nringlet);
}
BFsequence_sptr BFring_impl::begin_sequence(const char* name,
                                            BFoffset    time_tag,
                                            BFsize      header_size,
                                            const void* header,
                                            BFsize      nringlet,
                                            BFoffset    offset_from_head) {
	BF_ASSERT_EXCEPTION(name,                   BF_STATUS_INVALID_ARGUMENT);
	BF_ASSERT_EXCEPTION(header || !header_size, BF_STATUS_INVALID_ARGUMENT);
	lock_guard_type lock(_mutex);
	//unique_lock_type lock(_mutex);
	BF_ASSERT_EXCEPTION(nringlet <= _nringlet,  BF_STATUS_INVALID_ARGUMENT);
	// Cannot have any writes still open
	// TODO: Removed this since allowing writes independent of sequences
	//BF_ASSERT_EXCEPTION(_head == _reserve_head, BF_STATUS_INVALID_STATE);
	// Cannot have the previous sequence still open
	BF_ASSERT_EXCEPTION(_sequence_queue.empty() ||
	                    _sequence_queue.back()->is_finished(),
	                    BF_STATUS_INVALID_STATE);
	////_head         = round_up(_head, bfGetAlignment());
	//// Note: We force sequences to always begin on a multiple of the
	////         max contiguous span size.
	////         ACTUALLY, this complicates the packet-capture use-case
	////           and doesn't really contribute anything significant.
	////           It also adds a wait that is otherwise unnecessary
	////             and wastes space.
	//_head         = round_up(_head, _ghost_span);
	//_reserve_head = _head;
	//this->_pull_tail(lock); // Must be called after updating _reserve_head
	//BFoffset seq_begin = _reserve_head;
	BFoffset seq_begin = _head + offset_from_head;
	// Cannot have existing sequence with same name
	BF_ASSERT_EXCEPTION(_sequence_map.count(name)==0,              BF_STATUS_INVALID_ARGUMENT);
	BF_ASSERT_EXCEPTION(_sequence_time_tag_map.count(time_tag)==0, BF_STATUS_INVALID_ARGUMENT);
	BFsequence_sptr sequence(new BFsequence_impl(this, name, time_tag, header_size,
	                                             header, nringlet, seq_begin));
	if( _sequence_queue.size() ) {
		_sequence_queue.back()->set_next(sequence);
		//_sequence_condition.notify_all();
	}
	_sequence_queue.push(sequence);
	_sequence_condition.notify_all();
	if( !std::string(name).empty() ) {
		_sequence_map.insert(std::make_pair(std::string(name),sequence));
	}
	if( time_tag != BFoffset(-1) ) {
		_sequence_time_tag_map.insert(std::make_pair(time_tag,sequence));
	}
	return sequence;
}
void BFring_impl::open_sequence(BFsequence_sptr sequence,
                                BFbool          guarantee,
                                BFoffset*       guarantee_begin) {
	lock_guard_type lock(_mutex);
	// Check that the sequence is still within the ring
	BF_ASSERT_EXCEPTION(!sequence->is_finished() ||
	                    BFoffset(_head - sequence->end()) <= BFoffset(_head - _tail),
	                    BF_STATUS_INVALID_ARGUMENT);
	if( guarantee ) {
		if( BFoffset(_head - sequence->begin()) > BFoffset(_head - _tail) ) {
			// Sequence starts before tail
			*guarantee_begin = _tail;
		}
		else {
			*guarantee_begin = sequence->begin();
		}
		//_guarantees.insert(*guarantee_begin);
		this->_add_guarantee(*guarantee_begin);
	}
}
void BFring_impl::close_sequence(BFsequence_sptr sequence,
                                 BFbool          guarantee,
                                 BFoffset        guarantee_begin) {
	lock_guard_type lock(_mutex);
	if( guarantee ) {
		this->_remove_guarantee(guarantee_begin);
		//auto iter = _guarantees.find(guarantee_begin);
		//BF_ASSERT_EXCEPTION(iter != _guarantees.end(), BF_STATUS_INTERNAL_ERROR);
		//_guarantees.erase(iter);
	}
}
BFsequence_sptr BFring_impl::get_sequence(const char* name) {
	lock_guard_type lock(_mutex);
	BF_ASSERT_EXCEPTION(_sequence_map.count(name), BF_STATUS_INVALID_ARGUMENT);
	return _sequence_map.find(name)->second;
}
BFsequence_sptr BFring_impl::get_sequence_at(BFoffset time_tag) {
	lock_guard_type lock(_mutex);
	auto iter = _sequence_time_tag_map.upper_bound(time_tag);
	BF_ASSERT_EXCEPTION(iter != _sequence_time_tag_map.begin(),
	                    BF_STATUS_INVALID_ARGUMENT);
	return (--iter)->second;
}
BFsequence_sptr BFring_impl::get_latest_sequence() {
	unique_lock_type lock(_mutex);
	// Wait until a sequence has been opened or writing has ended
	_sequence_condition.wait(lock, [&]() {
			return !_sequence_queue.empty() || _writing_ended;
		});
	BF_ASSERT_EXCEPTION(!(_sequence_queue.empty() && !_writing_ended), BF_STATUS_INVALID_STATE);
	BF_ASSERT_EXCEPTION(!(_sequence_queue.empty() &&  _writing_ended), BF_STATUS_END_OF_DATA);
	//BF_ASSERT_EXCEPTION(!_writing_ended, BF_STATUS_END_OF_DATA);
	//BF_ASSERT_EXCEPTION(!_sequence_queue.empty(), BF_STATUS_INVALID_STATE);
	return _sequence_queue.back();
}
BFsequence_sptr BFring_impl::get_earliest_sequence() {
	unique_lock_type lock(_mutex);
	// Wait until a sequence has been opened or writing has ended
	_sequence_condition.wait(lock, [&]() {
			return !_sequence_queue.empty() || _writing_ended;
		});
	BF_ASSERT_EXCEPTION(!(_sequence_queue.empty() && !_writing_ended), BF_STATUS_INVALID_STATE);
	BF_ASSERT_EXCEPTION(!(_sequence_queue.empty() &&  _writing_ended), BF_STATUS_END_OF_DATA);
	//BF_ASSERT_EXCEPTION(!_writing_ended, BF_STATUS_END_OF_DATA);
	//BF_ASSERT_EXCEPTION(!_sequence_queue.empty(), BF_STATUS_INVALID_STATE);
	return _sequence_queue.front();
}

BFsequence_impl::BFsequence_impl(BFring      ring,
                                 const char* name,
                                 BFoffset    time_tag,
                                 BFsize      header_size,
                                 const void* header,
                                 BFsize      nringlet,
                                 BFoffset    begin)
	: _ring(ring), _name(name), _time_tag(time_tag), _nringlet(nringlet),
	  _begin(begin),
	  _end(BF_SEQUENCE_OPEN),
	  _header((const char*)header,
	          (const char*)header+header_size),
	  //_header(new header_type((const char*)header,
	  //                        (const char*)header+header_size)),
	  _next(nullptr) {
	//std::cout << "BEGIN SEQUENCE: " << _begin << std::endl;
	  }
void BFsequence_impl::finish(BFoffset offset_from_head) {
	BFring_impl::lock_guard_type lock(_ring->_mutex);
	// Cannot have any writes still open
	// TODO: Changed this since allowing writes independent of sequences
	//BF_ASSERT_EXCEPTION(_ring->_head == _ring->_reserve_head, BF_STATUS_INVALID_STATE);
	// Must have the sequence still open
	BF_ASSERT_EXCEPTION(!_ring->_sequence_queue.empty() &&
	                    !_ring->_sequence_queue.back()->is_finished(),
	                    BF_STATUS_INVALID_STATE);
	_end = _ring->_head + offset_from_head;
	_ring->_read_condition.notify_all();
	//std::cout << "END SEQUENCE: " << _end << std::endl;
}
void BFsequence_impl::set_next(BFsequence_sptr next) {
	_next = next;
}
BFsequence_sptr BFsequence_impl::get_next() const {
	BFring_impl::unique_lock_type lock(_ring->_mutex);
	// Wait until the next sequence has been opened or writing has ended
	_ring->_sequence_condition.wait(lock, [&]() {
			return ((bool)_next) || _ring->_writing_ended;
		});
	BF_ASSERT_EXCEPTION(_next, BF_STATUS_END_OF_DATA);
	return _next;
}

void BFring_impl::_pull_tail(unique_lock_type& lock) {
	// This pulls the tail along to satisfy the maximum span between
	//   _reserve_head and _tail. This must be done whenever _reserve_head
	//   is updated.
	// Note: This may cause still_valid() to return false, but the write
	//         condition will still wait for guaranteed reads before
	//         allowing data to actually be overwritten. This also means
	//         that guaranteed reads cannot "cover for" unguaranteed
	//         siblings that are too slow.
	BFoffset cur_span = _reserve_head - _tail;
	if( cur_span > _span ) {
		_tail += cur_span - _span;
		// Delete old sequences
		while( !_sequence_queue.empty() &&
		       //_sequence_queue.front()->_end != BFsequence_impl::BF_SEQUENCE_OPEN &&
		       _sequence_queue.front()->is_finished() &&
		       //_sequence_queue.front()->_end <= _tail ) {
		       BFoffset(_head - _sequence_queue.front()->_end) >= BFoffset(_head - _tail) ) {
		    if( !_sequence_queue.front()->_name.empty() ) {
			  _sequence_map.erase(_sequence_queue.front()->_name);
			}
		    if( _sequence_queue.front()->_time_tag != BFoffset(-1) ) {
			    _sequence_time_tag_map.erase(_sequence_queue.front()->_time_tag);
		    }
			//delete _sequence_queue.front();
			_sequence_queue.pop();
		}
	}
	// Wait for any relevant guarantees
	_write_condition.wait(lock, [&]() {
			return ((_guarantees.empty() ||
			         //_guarantees.begin()->first >= _tail) &&
			         BFoffset(_head - _get_earliest_guarantee()) <= BFoffset(_head - _tail)) &&
			        _nrealloc_pending == 0);
		});
}

void BFring_impl::reserve_span(BFsize size, BFoffset* begin, void** data) {
	unique_lock_type lock(_mutex);
	BF_ASSERT_EXCEPTION(size <= _ghost_span, BF_STATUS_INVALID_ARGUMENT);
	
	*begin = _reserve_head;
	_reserve_head += size;
	this->_pull_tail(lock); // Must be called after updating _reserve_head
	/*
	_write_condition.wait(lock, [&]() {
			return ((_guarantees.empty() ||
			         //_guarantees.begin()->first >= _tail) &&
			         BFoffset(_head - _get_earliest_guarantee()) <= BFoffset(_head - _tail)) &&
			        _nrealloc_pending == 0);
		});
	*/
	++_nwrite_open;
	*data = _buf_pointer(*begin);
}
void BFring_impl::commit_span(BFoffset begin, BFsize reserve_size, BFsize commit_size) {
	unique_lock_type lock(_mutex);
	_ghost_write(begin, commit_size);

	// TODO: Refactor/tidy this function a bit

	// Note: This allows unused open blocks to be 'cancelled' if they
	//         are closed in reverse order.
	if( commit_size == 0 &&
	    _reserve_head == begin + reserve_size ) {
		// This is the last-opened block so we can 'cancel' it by pulling back
		//   the reserve head.
		_reserve_head = begin;
		--_nwrite_open;
		_realloc_condition.notify_all();
		return;
	}
	
	// Wait until this block is at the head
	// Note: This allows write blocks to be closed out of order,
	//         in which case they will block here until they are
	//         in order (i.e., they will automatically synchronise).
	//         This is useful for multithreading with OpenMP
	//std::cout << "(1) begin, head, rhead: " << begin << ", " << _head << ", " << _reserve_head << std::endl;
	_write_close_condition.wait(lock, [&]() {
			return (begin == _head);
		});
	_write_close_condition.notify_all();
	
	if( _reserve_head == _head + reserve_size ) {
		// This is the front-most wspan, so we can pull back
		//   the reserve head if commit_size < size.
		_reserve_head = _head + commit_size;
	}
	else if( commit_size < reserve_size ) {
		// There are reservations in front of this one, so we
		//   are not allowed to commit less than size.
		// TODO: How to deal with error here?
		//std::cout << "BFRING ERROR: Must commit whole wspan when other spans are reserved" << std::endl;
		//return;
		BF_ASSERT_EXCEPTION(false, BF_STATUS_INVALID_STATE);
	}
	_head += commit_size;
	
	_read_condition.notify_all();
	--_nwrite_open;
	_realloc_condition.notify_all();
	//std::cout << "(2) begin, head, rhead: " << begin << ", " << _head << ", " << _reserve_head << std::endl;
}

BFwspan_impl::BFwspan_impl(//BFwsequence sequence,
                           BFring      ring,
                           BFsize      size)
	: //BFspan_impl(sequence->sequence(), size),
	  BFspan_impl(ring, size),
	//_sequence(sequence),
	  _begin(0),
	  _commit_size(size), _data(nullptr) {
	this->ring()->reserve_span(size, &_begin, &_data);
}
BFwspan_impl* BFwspan_impl::commit(BFsize size) {
	BF_ASSERT_EXCEPTION(size <= this->size(), BF_STATUS_INVALID_ARGUMENT);
	_commit_size = size;
	return this;
}
BFwspan_impl::~BFwspan_impl() {
	this->ring()->commit_span(_begin, this->size(), _commit_size);
}

void BFring_impl::acquire_span(BFrsequence rsequence,
                               BFoffset    offset,
                               BFsize*     size_,
                               BFoffset*   begin_,
                               void**      data_) {
	BF_ASSERT_EXCEPTION(rsequence,             BF_STATUS_INVALID_HANDLE);
	BF_ASSERT_EXCEPTION(size_,                 BF_STATUS_INVALID_POINTER);
	BF_ASSERT_EXCEPTION(begin_,                BF_STATUS_INVALID_POINTER);
	BF_ASSERT_EXCEPTION(data_,                 BF_STATUS_INVALID_POINTER);
	BFsequence_sptr sequence = rsequence->sequence();
	unique_lock_type lock(_mutex);
	BF_ASSERT_EXCEPTION(*size_ <= _ghost_span, BF_STATUS_INVALID_ARGUMENT);
	
	// Check that the requested span is within the sequence
	//std::cout << "acquire_span(offset=" << offset << ", size_=" << *size_ << ")" << std::endl;
	//std::cout << "  sequence size: " << sequence->end()-sequence->begin() << std::endl;
	BF_ASSERT_EXCEPTION(!sequence->is_finished() ||
	                    offset < sequence->end()-sequence->begin(),
	                    BF_STATUS_END_OF_DATA);
	
	BFoffset begin = sequence->begin() + offset;
	*begin_ = begin;
	
	// **TODO: Does this break if begin > _head?
	//           This has implications throughout this function
	/*
	BF_DEBUG_PRINT(_head);
	BF_DEBUG_PRINT(_tail);
	BF_DEBUG_PRINT(begin);
	BF_DEBUG_PRINT(sequence->begin());
	if( sequence->is_finished() ) {
		BF_DEBUG_PRINT(sequence->end());
	}
	else {
		BF_DEBUG_PRINT(sequence->is_finished());
	}
	BF_DEBUG_PRINT(BFdelta(_head - begin));
	BF_DEBUG_PRINT(BFdelta(_head - _tail));
	BF_DEBUG_PRINT(BFdelta(_head - begin) <= BFdelta(_head - _tail));
	BF_ASSERT_EXCEPTION(BFdelta(_head - begin) <= BFdelta(_head - _tail),
	                    // TODO: BF_STATUS_UNAVAILABLE?
	                    BF_STATUS_INVALID_ARGUMENT);
	*/
	BFsize size = *size_;
	_read_condition.wait(lock, [&]() {
			return ((BFdelta(_head - begin) >= BFdelta(size) ||
			         sequence->is_finished()) &&
			        _nrealloc_pending == 0);
		});
	// Check that the requested span is within the sequence
	BF_ASSERT_EXCEPTION(!sequence->is_finished() ||
	                    offset < sequence->end() - sequence->begin(),
	                    BF_STATUS_END_OF_DATA);
	if( sequence->is_finished() ) {
		size = std::min(size, BFsize(sequence->end()-begin));
	}
	*size_ = size;
	/*
	// TODO: The size>0 condition here avoids returning an empty last
	//         span, but we need to think more about how the user should
	//         be notified of the end of the sequence.
	BF_ASSERT_EXCEPTION(size > 0 &&
	                    BFdelta(_head - begin) <= BFdelta(_head - _tail),
	                    // TODO: BF_STATUS_UNAVAILABLE?
	                    BF_STATUS_INVALID_ARGUMENT);
	*/
	++_nread_open;
	_ghost_read(begin, size);
	*data_ = _buf_pointer(begin);
}
void BFring_impl::release_span(BFrsequence sequence,
                               BFoffset    offset,
                               BFsize      size) {
	unique_lock_type lock(_mutex);
	if( sequence->guaranteed() ) {
		// Move the guarantee to the end of this span
		this->_remove_guarantee(sequence->guarantee_begin());
		//auto iter = _guarantees.find(sequence->guarantee_begin());
		//BF_ASSERT_EXCEPTION(iter != _guarantees.end(), BF_STATUS_INTERNAL_ERROR);
		//_guarantees.erase(iter);
		BFoffset new_begin = offset + size;
		//_guarantees.insert(new_begin);
		this->_add_guarantee(new_begin);
		sequence->set_guarantee_begin(new_begin);
	}
	--_nread_open;
	_realloc_condition.notify_all();
}

BFrspan_impl::BFrspan_impl(BFrsequence sequence,
                           BFoffset    offset,
                           BFsize      requested_size)
	: //BFspan_impl(sequence->sequence(), requested_size),
	  BFspan_impl(sequence->ring(), requested_size),
	  _sequence(sequence), _begin(0),
	  _data(nullptr) {
	BFsize returned_size = requested_size;
	this->ring()->acquire_span(sequence, offset, &returned_size, &_begin, &_data);
	this->set_base_size(returned_size);
}
BFrspan_impl::~BFrspan_impl() {
	this->ring()->release_span(_sequence, _begin, this->size());
}
