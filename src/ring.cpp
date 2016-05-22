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

// TODO: Add BF_TRY( ) to destructors too to ease debugging

#include <bifrost/ring.h>
#include "ring_impl.hpp"
#include "assert.hpp"

BFstatus bfRingCreate(BFring* ring, BFenum space) {
	BF_ASSERT(ring, BF_STATUS_INVALID_POINTER);
	BF_TRY(*ring = new BFring_impl(space),
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
	BF_TRY(ring->resize(max_contiguous_span,
	                    max_total_size,
	                    max_ringlets),
	       BF_NO_OP);
}
BFstatus bfRingGetSpace(BFring ring, BFspace* space) {
	BF_ASSERT(ring,  BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(space, BF_STATUS_INVALID_POINTER);
	BF_TRY(*space = ring->space(), BF_NO_OP);
}
//BFsize   bfRingGetNRinglet(BFring ring) {
//	BF_ASSERT(ring, 0);
//	return ring->nringlet();
//}
BFstatus bfRingLock(BFring ring) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY(ring->lock(), BF_NO_OP);
}
BFstatus bfRingUnlock(BFring ring) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY(ring->unlock(), BF_NO_OP);
}
BFstatus bfRingLockedGetData(BFring ring, void** data) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY(*data = ring->locked_data(),
	       *data = 0);
}
BFstatus bfRingLockedGetContiguousSpan(BFring ring, BFsize* size) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY(*size = ring->locked_contiguous_span(),
	       *size = 0);
}
BFstatus bfRingLockedGetTotalSpan(BFring ring, BFsize* size) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY(*size = ring->locked_total_span(),
	       *size = 0);
}
BFstatus bfRingLockedGetNRinglet(BFring ring, BFsize* n) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY(*n = ring->locked_nringlet(),
	       *n = 0);
}
BFstatus   bfRingLockedGetStride(BFring ring, BFsize* size) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY(*size = ring->locked_stride(),
	       *size = 0);
}

BFstatus bfRingBeginWriting(BFring ring) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY(ring->begin_writing(), BF_NO_OP);
}
BFstatus bfRingEndWriting(BFring ring) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY(ring->end_writing(), BF_NO_OP);
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
	BF_TRY(*sequence = new BFwsequence_impl(ring, name, time_tag, header_size, header,
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
	BF_TRY(*sequence = new BFrsequence_impl(ring->get_sequence(name),
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
	BF_TRY(*sequence = new BFrsequence_impl(ring->get_sequence_at(time_tag),
	                                        guarantee),
	       *sequence = 0);
}
BFstatus bfRingSequenceOpenLatest(BFrsequence* sequence,
                                  BFring       ring,
                                  BFbool       guarantee) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(ring,     BF_STATUS_INVALID_HANDLE);
	BF_TRY(*sequence = new BFrsequence_impl(ring->get_latest_sequence(),
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
	BF_TRY(*sequence = new BFrsequence_impl(ring->get_earliest_sequence(),
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
	BF_TRY(sequence->increment_to_next(), BF_NO_OP);
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
	BF_TRY(*sequence = new BFrsequence_impl(*existing),
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
	BF_TRY(*ring = sequence->ring(),
	       *ring = 0);
}
BFstatus bfRingSequenceGetName(BFsequence sequence, char const** name) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(name,     BF_STATUS_INVALID_POINTER);
	BF_TRY(*name = sequence->name(),
	       *name = 0);
}
BFstatus bfRingSequenceGetTimeTag(BFsequence sequence, BFoffset* time_tag) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(time_tag, BF_STATUS_INVALID_POINTER);
	BF_TRY(*time_tag = sequence->time_tag(),
	       *time_tag = 0);
}
BFstatus bfRingSequenceGetHeader(BFsequence sequence, void const** hdr) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(hdr,      BF_STATUS_INVALID_POINTER);
	BF_TRY(*hdr = sequence->header(),
	       *hdr = 0);
}
BFstatus bfRingSequenceGetHeaderSize(BFsequence sequence, BFsize* size) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(size,     BF_STATUS_INVALID_POINTER);
	BF_TRY(*size = sequence->header_size(),
	       *size = 0);
}
BFstatus bfRingSequenceGetNRinglet(BFsequence sequence, BFsize* n) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(n,        BF_STATUS_INVALID_POINTER);
	BF_TRY(*n = sequence->nringlet(),
	       *n = 0);
}

BFstatus   bfRingSpanReserve(BFwspan*    span,
                             //BFwsequence sequence,
                             BFring      ring,
                             BFsize      size) {
	BF_ASSERT(span,     BF_STATUS_INVALID_POINTER);
	//BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY(*span = new BFwspan_impl(//sequence,
	                                ring,
	                                size),
	       *span = 0);
}
// TODO: Separate setsize/shrink vs. commit methods?
BFstatus   bfRingSpanCommit(BFwspan span,
                            BFsize  size) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_TRY(delete span->commit(size), BF_NO_OP);
}

BFstatus   bfRingSpanAcquire(BFrspan*    span,
                             BFrsequence sequence,
                             BFoffset    offset,
                             BFsize      size) {
	BF_ASSERT(span,     BF_STATUS_INVALID_POINTER);
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_TRY(*span = new BFrspan_impl(sequence, offset, size),
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
	BF_TRY(*val = span->ring(),
	       *val = 0);
}
BFstatus bfRingSpanGetData(BFspan span, void** data) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(data, BF_STATUS_INVALID_POINTER);
	BF_TRY(*data = span->data(),
	       *data = 0);
}
BFstatus bfRingSpanGetSize(BFspan span, BFsize* val) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(val,  BF_STATUS_INVALID_POINTER);
	BF_TRY(*val = span->size(),
	       *val = 0);
}
BFstatus bfRingSpanGetStride(BFspan span, BFsize* val) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(val,  BF_STATUS_INVALID_POINTER);
	BF_TRY(*val = span->stride(),
	       *val = 0);
}
BFstatus bfRingSpanGetOffset(BFspan span, BFsize* val) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(val,  BF_STATUS_INVALID_POINTER);
	BF_TRY(*val = span->offset(),
	       *val = 0);
}
BFstatus bfRingSpanGetNRinglet(BFspan span, BFsize* val) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(val,  BF_STATUS_INVALID_POINTER);
	BF_TRY(*val = span->nringlet(),
	       *val = 0);
}
