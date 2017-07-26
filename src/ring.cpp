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

// TODO: Consider adding Add BF_TRY( ) to destructors too to ease debugging

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
BFstatus bfRingSetAffinity(BFring ring, int  core) {
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(core >= -1, BF_STATUS_INVALID_ARGUMENT);
	BF_ASSERT(BF_NUMA_ENABLED, BF_STATUS_UNSUPPORTED);
	BF_TRY_RETURN(ring->set_core(core));
}
BFstatus bfRingGetAffinity(BFring ring, int* core) {
	BF_ASSERT(ring,  BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(core,  BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN(*core = ring->core());
}
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
	                   *sequence = 0);
}
BFstatus    bfRingSequenceEnd(BFwsequence sequence,
                              BFoffset    offset_from_head) {

	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	sequence->set_end_offset_from_head(offset_from_head);
	delete sequence;
	return BF_STATUS_SUCCESS;
}
BFstatus bfRingSequenceOpen(BFrsequence* sequence,
                            BFring       ring,
                            const char*  name,
                            BFbool       guarantee) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(ring,     BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(name,     BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*sequence = new BFrsequence_impl(
		                   BFrsequence_impl::by_name(ring, name, guarantee)),
	                   *sequence = 0);
}
BFstatus bfRingSequenceOpenAt(BFrsequence* sequence,
                              BFring       ring,
                              BFoffset     time_tag,
                              BFbool       guarantee) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(ring,     BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(time_tag!=BFoffset(-1), BF_STATUS_INVALID_ARGUMENT);
	BF_TRY_RETURN_ELSE(*sequence = new BFrsequence_impl(
		                   BFrsequence_impl::at(ring, time_tag, guarantee)),
	                   *sequence = 0);
}
BFstatus bfRingSequenceOpenLatest(BFrsequence* sequence,
                                  BFring       ring,
                                  BFbool       guarantee) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(ring,     BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*sequence = new BFrsequence_impl(
		                   BFrsequence_impl::earliest_or_latest(ring, guarantee, true)),
	                   *sequence = 0);
}
BFstatus bfRingSequenceOpenEarliest(BFrsequence* sequence,
                                    BFring       ring,
                                    BFbool       guarantee) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(ring,     BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*sequence = new BFrsequence_impl(
		                   BFrsequence_impl::earliest_or_latest(ring, guarantee, false)),
	                   *sequence = 0);
}
BFstatus bfRingSequenceNext(BFrsequence sequence) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(sequence->increment_to_next());
}
//BFstatus bfRingSequenceOpenSame(BFrsequence* sequence,
//                                BFrsequence  existing) {
//	BF_ASSERT(sequence, BF_STATUS_INVALID_POINTER);
//	//BF_TRY(*sequence = new BFrsequence_impl(existing->sequence(),
//	//                                        existing->guaranteed()),
//	BF_TRY_RETURN_ELSE(*sequence = new BFrsequence_impl(*existing),
//	                   *sequence = 0);
//}
BFstatus bfRingSequenceClose(BFrsequence sequence) {
	BF_ASSERT(sequence, BF_STATUS_INVALID_HANDLE);
	delete sequence;
	return BF_STATUS_SUCCESS;
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
                             BFring      ring,
                             BFsize      size) {
	BF_ASSERT(span,     BF_STATUS_INVALID_POINTER);
	BF_ASSERT(ring, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*span = new BFwspan_impl(ring,
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

// Returns in *val the number of bytes in the span that have been overwritten
//   at the time of the call (always zero for guaranteed sequences).
BFstatus   bfRingSpanGetSizeOverwritten(BFrspan span, BFsize* val) {
	BF_ASSERT(span, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(val,  BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*val = span->size_overwritten(),
	                   *val = 0);
}

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
