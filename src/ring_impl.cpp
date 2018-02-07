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

// TODO: Confirm that resizing is completely safe in any state

// Note: Due to potential wrapping, offsets should never be compared using <,<=,>,>=
//         Always compare positive differences between offsets instead
//           E.g., offset < tail --> _head-offset > _head-tail

// **TODO: Work out whether/how to do resize inside begin_sequence
//           ACTUALLY, put this on hold for now (try it out in ring.py first)
//             The reason is because it requires adding args to the API, and
//               the functionality may not be needed in some use-cases
//               (e.g., the udp capture code).
//             ACTUALLY ACTUALLY, the fact that nringlet is only relevant
//               when resizing during begin_sequence (i.e., it doesn't make
//               sense for readers to specify nringlet in a call to resize)
//               motivates making this change.
//         Work out whether/how to support independent specification of
//           buffer_factor.

#include "ring_impl.hpp"
#include "utils.hpp"
#include "assert.hpp"
#include <bifrost/memory.h>

#include <bifrost/cuda.h>
#include "cuda.hpp"

#if BF_NUMA_ENABLED
#include <numa.h>
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
		_ring->_realloc_condition.wait(_lock, [this]() {
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

BFring_impl::BFring_impl(const char* name, BFspace space)
	: _name(name), _space(space), _buf(nullptr),
	  _ghost_span(0), _span(0), _stride(0), _nringlet(0), _offset0(0),
	  _tail(0), _head(0), _reserve_head(0),
	  _ghost_dirty_beg(_ghost_span),
	  _writing_begun(false), _writing_ended(false), _eod(0),
	  _nread_open(0), _nwrite_open(0), _nrealloc_pending(0),
	  _core(-1), _size_log(std::string("rings/")+name) {

#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
	BF_ASSERT_EXCEPTION(space==BF_SPACE_SYSTEM       ||
	                    space==BF_SPACE_CUDA         ||
	                    space==BF_SPACE_CUDA_HOST    ||
	                    space==BF_SPACE_CUDA_MANAGED,
	                    BF_STATUS_INVALID_ARGUMENT);
#else
	BF_ASSERT_EXCEPTION(space==BF_SPACE_SYSTEM,
	                    BF_STATUS_INVALID_ARGUMENT);
#endif
	
	// Create the ProcLog entry for this ring
	_write_proclog_entry();
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
	new_span = std::max(new_span, bfGetAlignment());
	// **TODO: See if can avoid doing this, so that ghost-region memcpys can
	//           be avoided if user chooses gulp sizes in whole multiplies
	//           regardless of whether they are powers of two or not.
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
#if BF_NUMA_ENABLED
	if( _core != -1 ) {
		BF_ASSERT_EXCEPTION(numa_available() != -1, BF_STATUS_UNSUPPORTED);
		int node = numa_node_of_cpu(_core);
		BF_ASSERT_EXCEPTION(node != -1, BF_STATUS_INVALID_ARGUMENT);
		numa_tonode_memory(new_buf, new_nbyte, node);
	}
#endif
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
		//_ghost_dirty = true; // TODO: Is this the right thing to do?
		//_ghost_dirty_beg = new_ghost_span; // TODO: Is this the right thing to do?
		_ghost_dirty_beg = 0; // TODO: Is this the right thing to do?
		bfFree(_buf, _space);
		bfStreamSynchronize();
	}
	_buf        = new_buf;
	_ghost_span = new_ghost_span;
	_span       = new_span;
	_stride     = new_stride;
	_nringlet   = new_nringlet;
	
	// Update the ProcLog entry for this ring
	_write_proclog_entry();
}
void BFring_impl::begin_writing() {
	lock_guard_type lock(_mutex);
	BF_ASSERT_EXCEPTION(!_writing_begun, BF_STATUS_INVALID_STATE);
	BF_ASSERT_EXCEPTION(!_writing_ended, BF_STATUS_INVALID_STATE);
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
	if( buf_offset_beg < (BFoffset)_ghost_span ) {
		// The write touched the ghosted front of the buffer
		_ghost_dirty_beg = std::min(_ghost_dirty_beg, buf_offset_beg);
	}
}
void BFring_impl::_ghost_read(BFoffset offset, BFsize span) {
	BFoffset buf_offset_beg = _buf_offset(offset);
	BFoffset buf_offset_end = _buf_offset(offset + span);
	if( buf_offset_end < buf_offset_beg ) {
		// The read will enter the ghost region, so copy from the ghosted part
		buf_offset_end = std::min(buf_offset_end, (BFoffset)_ghost_span);
		BFsize dirty_span =
			std::max((BFdelta)buf_offset_end - (BFdelta)_ghost_dirty_beg,
			         BFdelta(0));
		this->_copy_to_ghost(_ghost_dirty_beg, dirty_span);
		// Note: This is actually _decreasing_ the amount that is marked dirty
		_ghost_dirty_beg += dirty_span;
	}
}
void BFring_impl::_copy_to_ghost(BFoffset buf_offset, BFsize span) {
	// Copy from the front of the buffer to the ghost region at the end
	bfMemcpy2D(_buf + (_span + buf_offset), _stride, _space,
	           _buf + buf_offset,           _stride, _space,
	           span, _nringlet);
	bfStreamSynchronize();
}
void BFring_impl::_copy_from_ghost(BFoffset buf_offset, BFsize span) {
	// Copy from the ghost region to the front of the buffer
	bfMemcpy2D(_buf + buf_offset,           _stride, _space,
	           _buf + (_span + buf_offset), _stride, _space,
	           span, _nringlet);
	bfStreamSynchronize();
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
	// Cannot have the previous sequence still open
	BF_ASSERT_EXCEPTION(_sequence_queue.empty() ||
	                    _sequence_queue.back()->is_finished(),
	                    BF_STATUS_INVALID_STATE);
	BFoffset seq_begin = _head + offset_from_head;
	// Cannot have existing sequence with same name
	BF_ASSERT_EXCEPTION(_sequence_map.count(name)==0,              BF_STATUS_INVALID_ARGUMENT);
	BF_ASSERT_EXCEPTION(_sequence_time_tag_map.count(time_tag)==0, BF_STATUS_INVALID_ARGUMENT);
	BFsequence_sptr sequence(new BFsequence_impl(this, name, time_tag, header_size,
	                                             header, nringlet, seq_begin));
	if( _sequence_queue.size() ) {
		_sequence_queue.back()->set_next(sequence);
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

BFsequence_sptr BFring_impl::_get_sequence_by_name(const char* name) {
	BF_ASSERT_EXCEPTION(_sequence_map.count(name), BF_STATUS_INVALID_ARGUMENT);
	return _sequence_map.find(name)->second;
}
BFsequence_sptr BFring_impl::open_sequence_by_name(const char* name,
                                                   bool with_guarantee,
                                                   std::unique_ptr<Guarantee>& guarantee) {
	// Note: Guarantee uses locks, so must be kept outside the lock scope here
	std::unique_ptr<Guarantee> scoped_guarantee;
	if( with_guarantee ) {
		// Ensure a guarantee is held while waiting for sequence to exist
		scoped_guarantee = new_guarantee(this);
	}
	unique_lock_type lock(_mutex);
	BFsequence_sptr sequence = this->_get_sequence_by_name(name);
	if( scoped_guarantee ) {
		// Move guarantee to start of sequence
		scoped_guarantee->move_nolock(
			this->_get_start_of_sequence_within_ring(sequence));
	}
	// Transfer ownership to the caller
	guarantee = std::move(scoped_guarantee);
	return sequence;
}

BFsequence_sptr BFring_impl::_get_sequence_at(BFoffset time_tag) {
	// Note: This function only works if time_tag resides within the buffer
	//         (or in its overwritten history) at the time of the call.
	//         There is no way for the function to know if a time_tag
	//           representing the future will actually fall within the current
	//           sequence or in a later one, and thus the returned sequence
	//           may turn out to be incorrect.
	//         If time_tag falls before the first sequence currently in the
	//           buffer, the function returns BF_STATUS_INVALID_ARGUMENT.
	//         TLDR; only use time_tag values representing times that have
	//           already happened, and be careful not to call this function
	//           before the very first sequence has been created.
	auto iter = _sequence_time_tag_map.upper_bound(time_tag);
	BF_ASSERT_EXCEPTION(iter != _sequence_time_tag_map.begin(),
	                    BF_STATUS_INVALID_ARGUMENT);
	return (--iter)->second;
}
BFsequence_sptr BFring_impl::open_sequence_at(BFoffset time_tag,
                                              bool with_guarantee,
                                              std::unique_ptr<Guarantee>& guarantee) {
	// Note: Guarantee uses locks, so must be kept outside the lock scope here
	std::unique_ptr<Guarantee> scoped_guarantee;
	if( with_guarantee ) {
		// Ensure a guarantee is held while waiting for sequence to exist
		scoped_guarantee = new_guarantee(this);
	}
	unique_lock_type lock(_mutex);
	BFsequence_sptr sequence = this->_get_sequence_at(time_tag);
	if( scoped_guarantee ) {
		// Move guarantee to start of sequence
		scoped_guarantee->move_nolock(
			this->_get_start_of_sequence_within_ring(sequence));
	}
	// Transfer ownership to the caller
	guarantee = std::move(scoped_guarantee);
	return sequence;
}
bool BFring_impl::_sequence_still_within_ring(BFsequence_sptr sequence) const {
	return (!sequence->is_finished() ||
	        BFoffset(_head - sequence->end()) <= BFoffset(_head - _tail));
}

BFsequence_sptr BFring_impl::_get_earliest_or_latest_sequence(unique_lock_type& lock, bool latest) const {
	// Wait until a sequence has been opened or writing has ended
	_sequence_condition.wait(lock, [this]() {
			return !_sequence_queue.empty() || _writing_ended;
		});
	BF_ASSERT_EXCEPTION(!(_sequence_queue.empty() && !_writing_ended), BF_STATUS_INVALID_STATE);
	BF_ASSERT_EXCEPTION(!(_sequence_queue.empty() &&  _writing_ended), BF_STATUS_END_OF_DATA);
	BFsequence_sptr sequence = (latest ?
	                            _sequence_queue.back() :
	                            _sequence_queue.front());
	// Check that the sequence is still within the ring
	BF_ASSERT_EXCEPTION(this->_sequence_still_within_ring(sequence),
	                    BF_STATUS_INVALID_ARGUMENT);
	return sequence;
}

BFoffset BFring_impl::_get_start_of_sequence_within_ring(BFsequence_sptr sequence) const {
	if( BFoffset(_head - sequence->begin()) > BFoffset(_head - _tail) ) {
		// Sequence starts before tail
		return _tail;
	} else {
		return sequence->begin();
	}
}

BFsequence_sptr BFring_impl::_get_next_sequence(BFsequence_sptr sequence,
                                                unique_lock_type& lock) const {
	// Wait until the next sequence has been opened or writing has ended
	_sequence_condition.wait(lock, [&]() {
			return ((bool)sequence->_next) || _writing_ended;
		});
	BF_ASSERT_EXCEPTION(sequence->_next, BF_STATUS_END_OF_DATA);
	return sequence->_next;
}

BFsequence_sptr BFring_impl::open_earliest_or_latest_sequence(bool with_guarantee,
                                                              std::unique_ptr<Guarantee>& guarantee,
                                                              bool latest) {
	// Note: Guarantee uses locks, so must be kept outside the lock scope here
	std::unique_ptr<Guarantee> scoped_guarantee;
	if( with_guarantee ) {
		// Ensure a guarantee is held while waiting for sequence to exist
		scoped_guarantee = new_guarantee(this);
	}
	unique_lock_type lock(_mutex);
	BFsequence_sptr sequence = this->_get_earliest_or_latest_sequence(lock, latest);
	if( scoped_guarantee ) {
		// Move guarantee to start of sequence
		scoped_guarantee->move_nolock(
			this->_get_start_of_sequence_within_ring(sequence));
	}
	// Transfer ownership to the caller
	guarantee = std::move(scoped_guarantee);
	return sequence;
}

void BFring_impl::increment_sequence_to_next(BFsequence_sptr& sequence,
                                             std::unique_ptr<Guarantee>& guarantee) {
	// Take ownership of the guarantee (if it exists)
	// Note: Guarantee uses locks, so must be kept outside the lock scope here
	std::unique_ptr<Guarantee> scoped_guarantee = std::move(guarantee);
	unique_lock_type lock(_mutex);
	//BFsequence_sptr next_sequence = this->_get_next_sequence(sequence, lock);
	sequence = this->_get_next_sequence(sequence, lock);
	if( scoped_guarantee ) {
		// Move the guarantee to the start of the new sequence
		scoped_guarantee->move_nolock(
			this->_get_start_of_sequence_within_ring(sequence));
	}
	// Return ownership of the guarantee
	guarantee = std::move(scoped_guarantee);
}

void BFring_impl::finish_sequence(BFsequence_sptr sequence,
                                  BFoffset offset_from_head) {
	lock_guard_type lock(_mutex);
	// Must have the sequence still open
	BF_ASSERT_EXCEPTION(!_sequence_queue.empty() &&
	                    !_sequence_queue.back()->is_finished(),
	                    BF_STATUS_INVALID_STATE);
	// This marks the sequence as finished
	sequence->_end = _head + offset_from_head;
	_read_condition.notify_all();
}

void BFring_impl::_write_proclog_entry() {
	char cinfo[32]="";
	#if BF_NUMA_ENABLED
	snprintf(cinfo, 31, "binding   : %i\n", _core);
	#endif
	_size_log.update("space     : %s\n"
	                 "%s"
	                 "alignment : %llu\n"
	                 "ghost     : %llu\n"
	                 "span      : %llu\n"
	                 "stride    : %llu\n"
	                 "nringlet  : %llu\n", 
	                 bfGetSpaceString(_space), cinfo, bfGetAlignment(), _span, _ghost_span, _stride, _nringlet);
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
	  _next(nullptr) {
}

void BFsequence_impl::set_next(BFsequence_sptr next) {
	_next = next;
}
bool BFring_impl::_advance_reserve_head(unique_lock_type& lock, BFsize size,
                                        bool nonblocking) {
	// This waits until all guarantees have caught up to the new valid
	//   buffer region defined by _reserve_head, and then pulls the tail
	//   along to ensure it is within a distance of _span from _reserve_head.
	
	// Note: By using _span, this correctly handles ring resizes that occur
	//         while waiting on the condition.
	// TODO: This enables guaranteed reads to "cover for" unguaranteed
	//         siblings that would be too slow on their own. Is this actually
	//         a problem, and if so is there any way around it?
	_reserve_head += size;
	auto postcondition_predicate = [this]() {
		return ((_guarantees.empty() ||
		         BFoffset(_reserve_head - _get_earliest_guarantee()) <= _span) &&
		        _nrealloc_pending == 0);
	};
	if( !nonblocking ) {
		_write_condition.wait(lock, postcondition_predicate);
	} else if( !postcondition_predicate() ) {
		// Revert and return failure
		_reserve_head -= size;
		return false;
	}
	
	BFoffset cur_span = _reserve_head - _tail;
	if( cur_span > _span ) {
		// Pull the tail
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
	return true;
}

void BFring_impl::reserve_span(BFsize size, BFoffset* begin, void** data,
                               bool nonblocking) {
	unique_lock_type lock(_mutex);
	BF_ASSERT_EXCEPTION(size <= _ghost_span, BF_STATUS_INVALID_ARGUMENT);
	*begin = _reserve_head;
	BF_ASSERT_EXCEPTION(this->_advance_reserve_head(lock, size, nonblocking),
	                    BF_STATUS_WOULD_BLOCK);
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
}

BFwspan_impl::BFwspan_impl(BFring ring,
                           BFsize size,
                           bool   nonblocking)
	: BFspan_impl(ring, size),
	  _begin(0),
	  _commit_size(size), _data(nullptr) {
	this->ring()->reserve_span(size, &_begin, &_data, nonblocking);
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
                               BFoffset    offset, // Relative to sequence beg
                               BFsize*     size_,
                               BFoffset*   begin_,
                               void**      data_) {
	BF_ASSERT_EXCEPTION(rsequence,             BF_STATUS_INVALID_HANDLE);
	BF_ASSERT_EXCEPTION(size_,                 BF_STATUS_INVALID_POINTER);
	BF_ASSERT_EXCEPTION(begin_,                BF_STATUS_INVALID_POINTER);
	BF_ASSERT_EXCEPTION(data_,                 BF_STATUS_INVALID_POINTER);
	// Cannot go back beyond the start of the sequence
	BF_ASSERT_EXCEPTION(offset >= 0,           BF_STATUS_INVALID_ARGUMENT);
	BFsequence_sptr sequence = rsequence->sequence();
	unique_lock_type lock(_mutex);
	BF_ASSERT_EXCEPTION(*size_ <= _ghost_span, BF_STATUS_INVALID_ARGUMENT);
	
	BFoffset requested_begin = sequence->begin() + offset;
	BFoffset requested_end   = requested_begin + *size_;
	
	// TODO: If this function fails, should the guarantee be left where it was?
	//         This would be straightforward to implement using a scoped
	//           guarantee.
	
	if( rsequence->guarantee() ) {
		BFoffset guarantee_begin = rsequence->guarantee()->offset();
		BFdelta distance_from_guarantee = BFdelta(requested_begin -
		                                          guarantee_begin);
		// Note: Triggered dumps may open a guaranteed sequence that has
		//         already been partially overwritten. In such cases, the
		//         user may reasonably request spans at the beginning of the
		//         sequence that actually lie outside of the guarantee
		//         (i.e., distance_from_guarantee < 0), and so an error should
		//         _not_ be returned in this scenario (just a zero-size span).
		if( distance_from_guarantee > 0 ) {
			// Move the guarantee forward to the beginning of this span to
			//   allow writers to make progress.
			rsequence->guarantee()->move_nolock(requested_begin);
		}
	}
	
	// This function returns whatever part of the requested span is available
	//   (meaning not overwritten and not past the end of the sequence).
	//   It will return a 0-length span if the requested span has been
	//     completely overwritten.
	// It throws BF_STATUS_END_OF_DATA if the requested span begins
	//   after the end of the sequence.
	
	// Wait until requested span has been written or sequence has ended
	_read_condition.wait(lock, [&]() {
			return ((BFdelta(_head         - std::max(requested_begin, _tail)) >=
			         BFdelta(requested_end - std::max(requested_begin, _tail)) ||
			         sequence->is_finished()) &&
			        _nrealloc_pending == 0);
		});
	
	// Constrain to what is in the buffer (i.e., what hasn't been overwritten)
	BFoffset begin = std::max(requested_begin, _tail);
	// Note: This results in size being 0 if the requested span has been
	//         completely overwritten.
	BFsize   size  = std::max(BFdelta(requested_end - begin), BFdelta(0));
	
	if( sequence->is_finished() ) {
		BF_ASSERT_EXCEPTION(begin < sequence->end(),
		                    BF_STATUS_END_OF_DATA);
		size = std::min(size, BFsize(sequence->end() - begin));
	}
	*begin_ = begin;
	*size_  = size;
	
	++_nread_open;
	_ghost_read(begin, size);
	*data_ = _buf_pointer(begin);
}
void BFring_impl::release_span(BFrsequence sequence,
                               BFoffset    begin,
                               BFsize      size) {
	unique_lock_type lock(_mutex);
	--_nread_open;
	_realloc_condition.notify_all();
}

BFrspan_impl::BFrspan_impl(BFrsequence sequence,
                           BFoffset    offset, // Relative to sequence beg
                           BFsize      requested_size)
	: BFspan_impl(sequence->ring(), requested_size),
	  _sequence(sequence), _begin(0),
	  _data(nullptr) {
	BFsize returned_size = requested_size;
	this->ring()->acquire_span(sequence, offset, &returned_size, &_begin, &_data);
	this->set_base_size(returned_size);
}
BFrspan_impl::~BFrspan_impl() {
	this->ring()->release_span(_sequence, _begin, this->size());
}
