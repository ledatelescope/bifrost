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

/*! \file ring.h
 *  \brief Ring buffer creation and management functions
 */

/*
TODO: New feature: mipmap-ringlets
        Store recursive 2x down-sampled rate versions in ringlets
        How to manage ghost regions?
      See if can avoid runtime memory allocations (i.e., all the new/delete)
*/

/* 
 * Ring:     A thread-safe circular memory buffer
 * Ringlets: Rings may be divided into multiple ringlets to enable time-fastest data ordering
 * Sequence: A logical range of data with an attached header
 *   Header:    A chunk of binary data attached to a sequence
 *   Name:      A string name uniquely identifying a sequence
 *   Time tag:  An integer number identifying the absolute time/position of a sequence
 *   Guarantee: Whether read access to a sequence is guaranteed or overwriteable
 * Span:     A physical range of data (contiguous memory)
 * 
 */

#ifndef BF_RING_H_INCLUDE_GUARD_
#define BF_RING_H_INCLUDE_GUARD_

#include <bifrost/common.h>
#include <bifrost/memory.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BFring_impl*        BFring;
typedef struct BFsequence_wrapper* BFsequence;
typedef struct BFrsequence_impl*   BFrsequence;
typedef struct BFwsequence_impl*   BFwsequence;
typedef struct BFspan_impl*        BFspan;
typedef struct BFrspan_impl*       BFrspan;
typedef struct BFwspan_impl*       BFwspan;

// TODO: bfCudaEnabled

// Ring
BFstatus bfRingCreate(BFring* ring, const char* name, BFspace space);
BFstatus bfRingDestroy(BFring ring);
/*! \p bfRingResize requests allocation of memory for the ring
 * 
 * \param ring             Handle of ring object
 * \param contiguous_bytes Max no. contiguous bytes to which access will be required
 * \param capacity_bytes   Max no. bytes required to be buffered (typically n*\p contiguous_bytes)
 * \param nringlet         Max no. ringlets required
 * \return One of the following error codes: \n
 * \p BIFROST_STATUS_SUCCESS, 
 * \note This function is thread-safe and can be called multiple times; reallocation
 * will occur only when necessary.
 */
BFstatus bfRingResize(BFring ring,
                      BFsize contiguous_bytes,
                      BFsize capacity_bytes,
                      BFsize nringlet);
BFstatus bfRingGetName(BFring ring, const char** name);
BFstatus bfRingGetSpace(BFring ring, BFspace* space);

/*! \p bfRingSetAffinity causes subsequent ring memory allocations to be bound
 *       to the NUMA node of the specified CPU core.
 * \param core Index of a CPU core on the desired NUMA node. A value of -1
 *          disabled NUMA affinity for subsequent memory allocations.
 */
BFstatus bfRingSetAffinity(BFring ring, int  core);
/*! \p bfRingGetAffinity returns the CPU core specified by a prior call to
 *     \p bfRingSetAffinity.
 * \param core Pointer to variable in which the CPU core index will be written.
 *        If \p bfRingSetAffinity has not been called, the variable will be
 *        set to a value of -1.
 */
BFstatus bfRingGetAffinity(BFring ring, int* core);

//BFsize   bfRingGetNRinglet(BFring ring);
// TODO: BFsize bfRingGetSizeBytes
// TODO: Method that returns tail,head,reserve_head plus all sequences' begin,end
//         OR, would a global lock plus query methods be better?
//       Are there other use-cases for lock/unlock?
//       Does the mutex need to be recursive to make these safe?
// TODO: Are these a good idea?
BFstatus bfRingLock(BFring ring);
BFstatus bfRingUnlock(BFring ring);
BFstatus bfRingLockedGetData(BFring ring, void** data);
BFstatus bfRingLockedGetContiguousSpan(BFring ring, BFsize* val);
BFstatus bfRingLockedGetTotalSpan(BFring ring, BFsize* val);
BFstatus bfRingLockedGetNRinglet(BFring ring, BFsize* val);
BFstatus bfRingLockedGetStride(BFring ring, BFsize* val);

// Note: These allow one to ensure that processing is completed before
//         the ring is destroyed. EndWriting effects an end to the
//         series of sequences.
BFstatus bfRingBeginWriting(BFring ring);
BFstatus bfRingEndWriting(BFring ring);
BFstatus bfRingWritingEnded(BFring ring, BFbool* writing_ended);

// Sequence write
// Note: \p name must either be unique among sequences, or be an empty string
// Note: \p time_tag should either be monotonically increasing with each
//         sequence, or be BFoffset(-1).
BFstatus bfRingSequenceBegin(BFwsequence* sequence,
                             BFring       ring,
                             const char*  name,
                             BFoffset     time_tag,
                             BFsize       header_size,
                             const void*  header,
                             BFsize       nringlet,
                             BFoffset     offset_from_head);
BFstatus bfRingSequenceEnd(BFwsequence sequence,
                           BFoffset    offset_from_head);

// Sequence read
BFstatus bfRingSequenceOpen(BFrsequence* sequence,
                            BFring       ring,
                            const char*  name,
                            BFbool       guarantee);
// Note: This function only works if time_tag resides within the buffer
//         (or in its overwritten history) at the time of the call.
//         If time_tag falls before the first sequence currently in the
//           buffer, the function returns BF_STATUS_INVALID_ARGUMENT.
//         If time_tag falls after the current head of the buffer,
//           the returned sequence may turn out not to be the one that
//           time_tag actually ends up falling into.
BFstatus bfRingSequenceOpenAt(BFrsequence* sequence,
                              BFring       ring,
                              BFoffset     time_tag,
                              BFbool       guarantee);
BFstatus bfRingSequenceOpenLatest(BFrsequence* sequence,
                                  BFring       ring,
                                  BFbool       guarantee);
BFstatus bfRingSequenceOpenEarliest(BFrsequence* sequence,
                                    BFring       ring,
                                    BFbool       guarantee);
//BFstatus bfRingSequenceOpenNext(BFrsequence* sequence, BFrsequence previous);
//BFstatus bfRingSequenceNext(BFrsequence* sequence);
BFstatus bfRingSequenceNext(BFrsequence sequence);
//BFstatus bfRingSequenceOpenSame(BFrsequence* sequence, BFrsequence existing);
BFstatus bfRingSequenceClose(BFrsequence sequence);

// Sequence common
BFstatus bfRingSequenceGetRing(BFsequence sequence, BFring* ring);
BFstatus bfRingSequenceGetName(BFsequence sequence, const char** name);
BFstatus bfRingSequenceGetTimeTag(BFsequence sequence, BFoffset* time_tag);
BFstatus bfRingSequenceGetHeader(BFsequence sequence, const void** hdr);
BFstatus bfRingSequenceGetHeaderSize(BFsequence sequence, BFsize* size);
BFstatus bfRingSequenceGetNRinglet(BFsequence sequence, BFsize* nringlet);

typedef struct BFsequence_info_ {
	BFring      ring;
	const char* name;
	BFoffset    time_tag;
	const void* header;
	BFsize      header_size;
	BFsize      nringlet;
} BFsequence_info;
// TODO: Implement this and remove the above individual functions
BFstatus bfRingSequenceGetInfo(BFsequence sequence, BFsequence_info* sequence_info);

// Write span
BFstatus bfRingSpanReserve(BFwspan*    span,
                           //BFwsequence sequence,
                           BFring      ring,
                           BFsize      size);
BFstatus bfRingSpanCommit(BFwspan span,
                          BFsize  size);
// Read span
BFstatus bfRingSpanAcquire(BFrspan*    span,
                           BFrsequence sequence,
                           BFoffset    offset,
                           BFsize      size);
BFstatus bfRingSpanRelease(BFrspan span);

// Returns in *val the number of bytes in the span that have been overwritten
//   at the time of the call (always zero for guaranteed sequences).
BFstatus bfRingSpanGetSizeOverwritten(BFrspan span, BFsize* val);
//BFbool bfRingSpanGood(BFrspan span); // true if span opened successfully
//BFstatus bfRingSpanGetSequence(BFspan span, BFrsequence* sequence);
// Any span
BFstatus bfRingSpanGetRing(BFspan span, BFring* data);
BFstatus bfRingSpanGetData(BFspan span, void** data);
BFstatus bfRingSpanGetSize(BFspan  span, BFsize* val);
BFstatus bfRingSpanGetStride(BFspan span, BFsize* val);
BFstatus bfRingSpanGetOffset(BFspan span, BFsize* val);
BFstatus bfRingSpanGetNRinglet(BFspan span, BFsize* val);

typedef struct BFspan_info_ {
	BFring      ring;
	void*       data;
	BFsize      size;
	BFsize      stride;
	BFsize      offset;
	BFsize      nringlet;
} BFspan_info;
BFstatus bfRingSpanGetInfo(BFspan span, BFspan_info* span_info);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_RING_H_INCLUDE_GUARD_
