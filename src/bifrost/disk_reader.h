/*
 * Copyright (c) 2017, The Bifrost Authors. All rights reserved.
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

#ifndef BF_DISK_READER_H_INCLUDE_GUARD_
#define BF_DISK_READER_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
	#endif
	
	#include <bifrost/ring.h>
	
	typedef struct BFdiskreader_impl* BFdiskreader;
	
	typedef int (*BFdiskreader_sequence_callback)(BFoffset, BFoffset, int, int, int,
	                                              void const**, size_t*);
	
	typedef enum BFdiskreader_status_ {
		BF_READ_STARTED,
		BF_READ_ENDED,
		BF_READ_CONTINUED,
		BF_READ_CHANGED,
		BF_READ_NO_DATA,
		BF_READ_INTERRUPTED,
		BF_READ_ERROR
	} BFdiskreader_status;
	
	BFstatus bfDiskReaderCreate(BFdiskreader* obj,
	                            const char*   format,
	                            int           fd,
	                            BFring        ring,
	                            BFsize        nsrc,
	                            BFsize        src0,
	                            BFsize        buffer_ntime,
	                            BFsize        slot_ntime,
	                            BFdiskreader_sequence_callback sequence_callback,
	                            int           core);
	BFstatus bfDiskReaderDestroy(BFdiskreader obj);
	BFstatus bfDiskReaderRead(BFdiskreader obj, BFdiskreader_status* result);
	BFstatus bfDiskReaderFlush(BFdiskreader obj);
	BFstatus bfDiskReaderEnd(BFdiskreader obj);
	
	#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_DISK_READER_H_INCLUDE_GUARD_
