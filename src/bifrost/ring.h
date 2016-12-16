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
BFstatus bfRingCreate(BFring* ring, BFspace space);
BFstatus bfRingDestroy(BFring ring);
BFstatus bfRingResize(BFring ring,
                      BFsize contiguous_span,
                      // TODO: Consider not using 'total' to avoid confusion
                      BFsize total_span,
                      BFsize nringlet);
BFstatus bfRingGetSpace(BFring ring, BFspace* space);

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

// Sequence write
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
BFstatus bfRingSequenceOpenSame(BFrsequence* sequence, BFrsequence existing);
BFstatus bfRingSequenceClose(BFrsequence sequence);

// Sequence common
BFstatus bfRingSequenceGetRing(BFsequence sequence, BFring* ring);
BFstatus bfRingSequenceGetName(BFsequence sequence, const char** name);
BFstatus bfRingSequenceGetTimeTag(BFsequence sequence, BFoffset* time_tag);
BFstatus bfRingSequenceGetHeader(BFsequence sequence, const void** hdr);
BFstatus bfRingSequenceGetHeaderSize(BFsequence sequence, BFsize* size);
BFstatus bfRingSequenceGetNRinglet(BFsequence sequence, BFsize* nringlet);

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

//BFstatus bfRingSpanClose(BFrspan span);
BFstatus bfRingSpanStillValid(BFrspan  span,
                              BFoffset offset,
                              BFbool*  valid); // true if span not overwritten beyond offset
//BFbool bfRingSpanGood(BFrspan span); // true if span opened successfully
BFstatus bfRingSpanGetSequence(BFspan span, BFrsequence* sequence);
// Any span
BFstatus bfRingSpanGetRing(BFspan span, BFring* data);
BFstatus bfRingSpanGetData(BFspan span, void** data);
BFstatus bfRingSpanGetSize(BFspan  span, BFsize* val);
BFstatus bfRingSpanGetStride(BFspan span, BFsize* val);
BFstatus bfRingSpanGetOffset(BFspan span, BFsize* val);
BFstatus bfRingSpanGetNRinglet(BFspan span, BFsize* val);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_RING_H_INCLUDE_GUARD_
