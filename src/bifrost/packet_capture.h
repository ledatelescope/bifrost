/*
 * Copyright (c) 2019, The Bifrost Authors. All rights reserved.
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

#ifndef BF_PACKET_CAPTURE_H_INCLUDE_GUARD_
#define BF_PACKET_CAPTURE_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

#include <bifrost/ring.h>

// Callback setup

typedef int (*BFpacketcapture_chips_sequence_callback)(BFoffset, int, int, int,
                                                       BFoffset*, void const**, size_t*);
typedef int (*BFpacketcapture_ibeam_sequence_callback)(BFoffset, int, int, int,
                                                       BFoffset*, void const**, size_t*);
typedef int (*BFpacketcapture_cor_sequence_callback)(BFoffset, BFoffset, int, int,
                                                     int, int, void const**, size_t*);
typedef int (*BFpacketcapture_vdif_sequence_callback)(BFoffset, BFoffset, int, int, int,
                                                     int, int, int, void const**, size_t*);
typedef int (*BFpacketcapture_tbn_sequence_callback)(BFoffset, BFoffset, int, int, 
                                                     int, void const**, size_t*);
typedef int (*BFpacketcapture_drx_sequence_callback)(BFoffset, BFoffset, int, int, int, 
                                                     int, void const**, size_t*);

typedef struct BFpacketcapture_callback_impl* BFpacketcapture_callback;

BFstatus bfPacketCaptureCallbackCreate(BFpacketcapture_callback* obj);
BFstatus bfPacketCaptureCallbackDestroy(BFpacketcapture_callback obj);
BFstatus bfPacketCaptureCallbackSetCHIPS(BFpacketcapture_callback obj,
                                         BFpacketcapture_chips_sequence_callback callback);
BFstatus bfPacketCaptureCallbackSetIBeam(BFpacketcapture_callback obj,
                                         BFpacketcapture_ibeam_sequence_callback callback);
BFstatus bfPacketCaptureCallbackSetCOR(BFpacketcapture_callback obj,
                                       BFpacketcapture_cor_sequence_callback callback);
BFstatus bfPacketCaptureCallbackSetVDIF(BFpacketcapture_callback obj,
                                        BFpacketcapture_vdif_sequence_callback callback);
BFstatus bfPacketCaptureCallbackSetTBN(BFpacketcapture_callback obj,
                                       BFpacketcapture_tbn_sequence_callback callback);
BFstatus bfPacketCaptureCallbackSetDRX(BFpacketcapture_callback obj,
                                       BFpacketcapture_drx_sequence_callback callback);

// Capture setup

typedef struct BFpacketcapture_impl* BFpacketcapture;

typedef enum BFpacketcapture_status_ {
        BF_CAPTURE_STARTED,
        BF_CAPTURE_ENDED,
        BF_CAPTURE_CONTINUED,
        BF_CAPTURE_CHANGED,
        BF_CAPTURE_NO_DATA,
        BF_CAPTURE_INTERRUPTED,
        BF_CAPTURE_ERROR
} BFpacketcapture_status;

BFstatus bfDiskReaderCreate(BFpacketcapture* obj,
                            const char*      format,
                            int              fd,
                            BFring           ring,
                            BFsize           nsrc,
                            BFsize           src0,
                            BFsize           buffer_ntime,
                            BFsize           slot_ntime,
                            BFpacketcapture_callback sequence_callback,
                            int              core);
BFstatus bfUdpCaptureCreate(BFpacketcapture* obj,
                            const char*      format,
                            int              fd,
                            BFring           ring,
                            BFsize           nsrc,
                            BFsize           src0,
                            BFsize           max_payload_size,
                            BFsize           buffer_ntime,
                            BFsize           slot_ntime,
                            BFpacketcapture_callback sequence_callback,
                            int              core);
BFstatus bfUdpSnifferCreate(BFpacketcapture* obj,
                            const char*      format,
                            int              fd,
                            BFring           ring,
                            BFsize           nsrc,
                            BFsize           src0,
                            BFsize           max_payload_size,
                            BFsize           buffer_ntime,
                            BFsize           slot_ntime,
                            BFpacketcapture_callback sequence_callback,
                            int              core);
BFstatus bfUdpVerbsCaptureCreate(BFpacketcapture* obj,
                                 const char*      format,
                                 int              fd,
                                 BFring           ring,
                                 BFsize           nsrc,
                                 BFsize           src0,
                                 BFsize           max_payload_size,
                                 BFsize           buffer_ntime,
                                 BFsize           slot_ntime,
                                 BFpacketcapture_callback sequence_callback,
                                 int              core);
BFstatus bfPacketCaptureDestroy(BFpacketcapture obj);
BFstatus bfPacketCaptureRecv(BFpacketcapture         obj,
                             BFpacketcapture_status* result);
BFstatus bfPacketCaptureFlush(BFpacketcapture obj);
BFstatus bfPacketCaptureSeek(BFpacketcapture obj,
                             BFoffset        offset,
                             BFiowhence      whence,
                             BFoffset*       position);
BFstatus bfPacketCaptureTell(BFpacketcapture obj,
                             BFoffset*       position);
BFstatus bfPacketCaptureEnd(BFpacketcapture obj);
// TODO: bfPacketCaptureGetXX

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_PACKET_CAPTURE_H_INCLUDE_GUARD_
