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

#ifndef BF_UDP_CAPTURE_H_INCLUDE_GUARD_
#define BF_UDP_CAPTURE_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

#include <bifrost/address.h>
#include <bifrost/ring.h>

typedef struct BFudpcapture_impl* BFudpcapture;

typedef int (*BFudpcapture_sequence_callback)(BFoffset, int, int, int,
                                              BFoffset*, void const**, size_t*);

typedef enum BFudpcapture_status_ {
	BF_CAPTURE_STARTED,
	BF_CAPTURE_ENDED,
	BF_CAPTURE_CONTINUED,
	BF_CAPTURE_CHANGED,
	BF_CAPTURE_NO_DATA,
	BF_CAPTURE_INTERRUPTED,
	BF_CAPTURE_ERROR
} BFudpcapture_status;

BFstatus bfUdpCaptureCreate(BFudpcapture* obj,
                            const char*   format,
                            int           fd,
                            BFring        ring,
                            BFsize        nsrc,
                            BFsize        src0,
                            BFsize        max_payload_size,
                            BFsize        buffer_ntime,
                            BFsize        slot_ntime,
                            BFudpcapture_sequence_callback sequence_callback,
                            int           core);
BFstatus bfUdpCaptureDestroy(BFudpcapture obj);
BFstatus bfUdpCaptureRecv(BFudpcapture obj, BFudpcapture_status* result);
BFstatus bfUdpCaptureFlush(BFudpcapture obj);
BFstatus bfUdpCaptureEnd(BFudpcapture obj);
// TODO: bfUdpCaptureGetXX

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_UDP_CAPTURE_H_INCLUDE_GUARD_
