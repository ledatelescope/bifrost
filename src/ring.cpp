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

// TODO: Add BF_TRY( ) to destructors too to ease debugging

#include <bifrost/ring.h>
#include "ring_impl.hpp"
#include "assert.hpp"
#include <cstring> // For ::memset

BFstatus bfRingCreate(BFring* ring, char const* name, BFspace space) {
	BF_ASSERT(ring, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(name, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*ring = new BFring_impl(name, space),
	                   *ring = 0);
}
BFstatus bfRingDestroy(BFring ring) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	delete ring;
	return BF_STATUS_SUCCESS;
}
BFstatus bfRingResize(BFring ring,
                      BFsize max_contiguous_span,
                      BFsize max_total_size,
                      BFsize max_ringlets) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(ring->resize(max_contiguous_span,
	                    max_total_size,
	                    max_ringlets));
}
BFstatus bfRingGetName(BFring ring, char const** name) {
	BF_ASSERT(ring,     BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(name,     BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*name = ring->name(),
	                   *name = 0);
}
BFstatus bfRingGetSpace(BFring ring, BFspace* space) {
	BF_ASSERT(ring,  BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(space, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN(*space = ring->space());
}
//BFsize   bfRingGetNRinglet(BFring ring) {
//	BF_ASSERT(ring, 0);
//	return ring->nringlet();
//}
BFstatus bfRingLock(BFring ring) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(ring->lock());
}
BFstatus bfRingUnlock(BFring ring) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(ring->unlock());
}
BFstatus bfRingLockedGetData(BFring ring, void** data) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*data = ring->locked_data(),
	                   *data = 0);
}
BFstatus bfRingLockedGetContiguousSpan(BFring ring, BFsize* size) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*size = ring->locked_contiguous_span(),
	                   *size = 0);
}
BFstatus bfRingLockedGetTotalSpan(BFring ring, BFsize* size) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*size = ring->locked_total_span(),
	                   *size = 0);
}
BFstatus bfRingLockedGetNRinglet(BFring ring, BFsize* n) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*n = ring->locked_nringlet(),
	                   *n = 0);
}
BFstatus   bfRingLockedGetStride(BFring ring, BFsize* size) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*size = ring->locked_stride(),
	                   *size = 0);
}

BFstatus bfRingBeginWriting(BFring ring) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(ring->begin_writing());
}
BFstatus bfRingEndWriting(BFring ring) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(ring->end_writing());
}
BFstatus bfRingWritingEnded(BFring ring, BFbool* writing_ended) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(*writing_ended = ring->writing_ended());
}

BFstatus    bfRingSequenceBegin(BFwsequence* sequence,
                                BFring       ring,
                                const char*  name,
                                BFoffset     time_tag,
                                BFsize       header_size,
                                const void*  header,
                                BFsize       nringlet,
                                BFoffset     offset_from_head) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(ring,     BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*sequence = new BFwsequence_impl(ring, name, time_tag, header_size, header,
	                                                    nringlet, offset_from_head),
	       //ring->begin_sequence(name, header_size, header, nringlet),
	                   *sequence = 0);
	/*
	//BF_TRY(*sequence = new BFsequence_sptr(new BFsequence_impl(ring, name, header_size, header, nringlet)),
	BF_TRY(*sequence = new BFwsequence_sptr(ring->begin_sequence(name, header_size, header, nringlet)),
	// TODO: Consider using a factory inside ring instead
	//BF_TRY(*sequence = new BFsequence_impl(ring, name, header_size, header, nringlet),
	       *sequence = 0);
	*/
}
BFstatus    bfRingSequenceEnd(BFwsequence sequence,
                              BFoffset    offset_from_head) {

	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	//(*sequence)->finish();
	sequence->set_end_offset_from_head(offset_from_head);
	delete sequence;
	return BF_STATUS_SUCCESS;
	/*
	// **TODO: Can't actually delete here, it must hang around inside ring
	//delete sequence;
	//BF_TRY((*sequence)->finish(), BF_NO_OP);
	(*sequence)->finish();
	delete sequence; // Delete the smart pointer
	return BF_STATUS_SUCCESS;
	*/
}

BFstatus bfRingSequenceOpen(BFrsequence* sequence,
                            BFring       ring,
                            const char*  name,
                            BFbool       guarantee) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(ring,     BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(name,     BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*sequence = new BFrsequence_impl(ring->get_sequence(name),
	                                                    guarantee),
	       //ring->get_sequence(name, guarantee),
	                   *sequence = 0);
	//BF_TRY(*sequence = new BFsequence_sptr(ring->get_sequence(name)),
	//       *sequence = 0);
}
BFstatus bfRingSequenceOpenAt(BFrsequence* sequence,
                              BFring       ring,
                              BFoffset     time_tag,
                              BFbool       guarantee) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(ring,     BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(time_tag!=BFoffset(-1), BF_STATUS_INVALID_ARGUMENT);
	BF_TRY_RETURN_ELSE(*sequence = new BFrsequence_impl(ring->get_sequence_at(time_tag),
	                                                    guarantee),
	                   *sequence = 0);
}
BFstatus bfRingSequenceOpenLatest(BFrsequence* sequence,
                                  BFring       ring,
                                  BFbool       guarantee) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(ring,     BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*sequence = new BFrsequence_impl(ring->get_latest_sequence(),
	                                                    guarantee),
	                   *sequence = 0);
	//BF_TRY(*sequence = new BFsequence_sptr(ring->get_latest_sequence()),
	//       *sequence = 0);
}
BFstatus bfRingSequenceOpenEarliest(BFrsequence* sequence,
                                    BFring       ring,
                                    BFbool       guarantee) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(ring,     BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*sequence = new BFrsequence_impl(ring->get_earliest_sequence(),
	                                                    guarantee),
	                   *sequence = 0);
	//BF_TRY(*sequence = new BFsequence_sptr(ring->get_earliest_sequence()),
	//       *sequence = 0);
}
//BFstatus bfRingSequenceOpenNext(BFrsequence* sequence,
//                                BFrsequence  previous) {
BFstatus bfRingSequenceNext(BFrsequence sequence) {
	//BF_ASSERT( sequence, BF_STATUS_INVALID_POINTER);
	//BF_ASSERT(*sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(sequence->increment_to_next());
	/*
	// TODO: This is a bit ugly
	//         It arose from the need to close the previous seq before opening
	//           the next one so that the previous guarantee doesn't block
	//           writers while it's waiting for next.
	BFrsequence previous = *sequence;
	BFrsequence next;
	bool previous_guaranteed;
	BF_TRY(previous_guaranteed = previous->guaranteed();
	       previous->close(); // Just removes the guarantee
	       next = new BFrsequence_impl(previous->get_next(),
	                                   previous_guaranteed),
	       *sequence = 0);
	BFstatus stat = bfRingSequenceClose(previous);
	if( stat != BF_STATUS_SUCCESS ) {
		return stat;
	}
	*sequence = next;
	return BF_STATUS_SUCCESS;
	*/
	//BF_TRY(sequence->close();
	//       sequence->get_next()
	//BF_TRY(*sequence = new BFrsequence_impl(previous->get_next(),
	//                                        previous->guaranteed()),
	//       *sequence = 0);
	/*
	// TODO: Check for errors
	BFsequence new_sequence = new BFsequence_sptr((**sequence)->get_next());
	delete *sequence; // Delete the smart pointer
	*sequence = new_sequence;
	return BF_STATUS_SUCCESS;
	*/
}
BFstatus bfRingSequenceOpenSame(BFrsequence* sequence,
                                BFrsequence  existing) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_POINTER);
	//BF_TRY(*sequence = new BFrsequence_impl(existing->sequence(),
	//                                        existing->guaranteed()),
	BF_TRY_RETURN_ELSE(*sequence = new BFrsequence_impl(*existing),
	                   *sequence = 0);
}
BFstatus bfRingSequenceClose(BFrsequence sequence) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	//sequence->close();
	delete sequence;
	return BF_STATUS_SUCCESS;
	//delete sequence; // Delete the smart pointer
	//return BF_STATUS_SUCCESS;
}

BFstatus    bfRingSequenceGetRing(BFsequence sequence, BFring* ring) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(ring,     BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*ring = sequence->ring(),
	                   *ring = 0);
}
BFstatus bfRingSequenceGetName(BFsequence sequence, char const** name) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(name,     BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*name = sequence->name(),
	                   *name = 0);
}
BFstatus bfRingSequenceGetTimeTag(BFsequence sequence, BFoffset* time_tag) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(time_tag, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*time_tag = sequence->time_tag(),
	                   *time_tag = 0);
}
BFstatus bfRingSequenceGetHeader(BFsequence sequence, void const** hdr) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(hdr,      BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*hdr = sequence->header(),
	                   *hdr = 0);
}
BFstatus bfRingSequenceGetHeaderSize(BFsequence sequence, BFsize* size) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(size,     BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*size = sequence->header_size(),
	                   *size = 0);
}
BFstatus bfRingSequenceGetNRinglet(BFsequence sequence, BFsize* n) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(n,        BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*n = sequence->nringlet(),
	                   *n = 0);
}

BFstatus   bfRingSpanReserve(BFwspan*    span,
                             //BFwsequence sequence,
                             BFring      ring,
                             BFsize      size) {
	BF_ASSERT(span,     BF_STATUS_INVALID_POINTER);
	//BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*span = new BFwspan_impl(//sequence,
	                                            ring,
	                                            size),
	                   *span = 0);
}
// TODO: Separate setsize/shrink vs. commit methods?
BFstatus   bfRingSpanCommit(BFwspan span,
                            BFsize  size) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(delete span->commit(size));
}

BFstatus   bfRingSpanAcquire(BFrspan*    span,
                             BFrsequence sequence,
                             BFoffset    offset,
                             BFsize      size) {
	BF_ASSERT(span,     BF_STATUS_INVALID_POINTER);
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*span = new BFrspan_impl(sequence, offset, size),
	                   *span = 0);
}
BFstatus   bfRingSpanRelease(BFrspan span) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	delete span;
	return BF_STATUS_SUCCESS;
}

/*
BFstatus bfRingSpanOpen(BFrspan*   span,
                        BFsequence sequence,
                        BFoffset   offset,
                        BFsize     size,
                        BFbool     guarantee);
BFstatus   bfRingSpanAdvance(BFrspan  span,
                             BFdelta  delta,
                             BFsize   size,
                             BFbool   guarantee);
BFstatus   bfRingSpanClose(BFrspan span) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	delete span;
	return BF_STATUS_SUCCESS;
}
BFstatus   bfRingSpanStillValid(BFrspan  span,
                                BFoffset offset,
                                BFbool*  valid); // true if span not overwritten beyond offset
*/
/*
BFstatus bfRingSpanGetSequence(BFspan span, BFsequence* sequence) {
	BF_ASSERT(span,     BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(sequence, BF_STATUS_INVALID_POINTER);
	BF_TRY(*sequence = span->sequence().get(),
	       *sequence = 0);
}
*/
BFstatus bfRingSpanGetRing(BFspan span, BFring* val) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(val,  BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*val = span->ring(),
	                   *val = 0);
}
BFstatus bfRingSpanGetData(BFspan span, void** data) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(data, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*data = span->data(),
	                   *data = 0);
}
BFstatus bfRingSpanGetSize(BFspan span, BFsize* val) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(val,  BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*val = span->size(),
	                   *val = 0);
}
BFstatus bfRingSpanGetStride(BFspan span, BFsize* val) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(val,  BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*val = span->stride(),
	                   *val = 0);
}
BFstatus bfRingSpanGetOffset(BFspan span, BFsize* val) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(val,  BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*val = span->offset(),
	                   *val = 0);
}
BFstatus bfRingSpanGetNRinglet(BFspan span, BFsize* val) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(val,  BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*val = span->nringlet(),
	                   *val = 0);
}
BFstatus bfRingSpanGetInfo(BFspan span, BFspan_info* span_info) {
	BF_ASSERT(span,      BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(span_info, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(span_info->ring     = span->ring();
	                   span_info->data     = span->data();
	                   span_info->size     = span->size();
	                   span_info->stride   = span->stride();
	                   span_info->offset   = span->offset();
	                   span_info->nringlet = span->nringlet(),
	                   ::memset(span_info, 0, sizeof(BFspan_info)));
}
