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

#ifndef BF_PACKET_FORMATS_H_INCLUDE_GUARD_
#define BF_PACKET_FORMATS_H_INCLUDE_GUARD_

enum {
	JUMBO_FRAME_SIZE = 9000,
	DRX_FRAME_SIZE   = 4128,
	TBN_FRAME_SIZE   = 1048
};

#pragma pack(1)
struct chips_hdr_type {
	uint8_t  roach;    // Note: 1-based
	uint8_t  gbe;      // (AKA tuning)
	uint8_t  nchan;    // 109
	uint8_t  nsubband; // 11
	uint8_t  subband;  // 0-11
	uint8_t  nroach;   // 16
	// Note: Big endian
	uint16_t chan0;    // First chan in packet
	uint64_t seq;      // Note: 1-based
};

#pragma pack(1)
struct drx_hdr_type {
	uint32_t sync_word;
	uint32_t frame_count_word;
	uint32_t seconds_count;
	uint16_t decimation;
	uint16_t time_offset;
	uint64_t time_tag;
	uint32_t tuning_word;
	uint32_t flags;
};

#pragma pack(1)
struct tbn_hdr_type {
	uint32_t sync_word;
	uint32_t frame_count_word;
	uint32_t tuning_word;
	uint16_t tbn_id;
	uint16_t gain;
	uint64_t time_tag;
};

#endif // BF_PACKET_FORMATS_H_INCLUDE_GUARD_