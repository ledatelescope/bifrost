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

#ifndef BF_DATA_CAPTURE_H_INCLUDE_GUARD_
#define BF_DATA_CAPTURE_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

#include <bifrost/address.h>
#include <bifrost/ring.h>

typedef struct BFdatacapture_impl* BFdatacapture;

typedef enum BFdatacapture_status_ {
        BF_CAPTURE_STARTED,
        BF_CAPTURE_ENDED,
        BF_CAPTURE_CONTINUED,
        BF_CAPTURE_CHANGED,
        BF_CAPTURE_NO_DATA,
        BF_CAPTURE_INTERRUPTED,
        BF_CAPTURE_ERROR
} BFdatacapture_status;

BFstatus bfDataCaptureDestroy(BFdatacapture obj);
BFstatus bfDataCaptureRecv(BFdatacapture obj, BFdatacapture_status* result);
BFstatus bfDataCaptureFlush(BFdatacapture obj);
BFstatus bfDataCaptureEnd(BFdatacapture obj);
// TODO: bfDataCaptureGetXX

typedef int (*BFdatacapture_chips_sequence_callback)(BFoffset, int, int, int,
                                                    BFoffset*, void const**, size_t*);
typedef int (*BFdatacapture_tbn_sequence_callback)(BFoffset, BFoffset, int, int, 
                                                   void const**, size_t*);
typedef int (*BFdatacapture_drx_sequence_callback)(BFoffset, BFoffset, int, int, int, 
                                                   void const**, size_t*);

typedef struct BFdatacapture_callback_impl* BFdatacapture_callback;

BFstatus bfDataCaptureCallbackCreate(BFdatacapture_callback* obj);
BFstatus bfDataCaptureCallbackDestroy(BFdatacapture_callback obj);
BFstatus bfDataCaptureCallbackSetCHIPS(BFdatacapture_callback obj, BFdatacapture_chips_sequence_callback callback);
BFstatus bfDataCaptureCallbackSetTBN(BFdatacapture_callback obj, BFdatacapture_tbn_sequence_callback callback);
BFstatus bfDataCaptureCallbackSetDRX(BFdatacapture_callback obj, BFdatacapture_drx_sequence_callback callback);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_DATA_CAPTURE_H_INCLUDE_GUARD_
