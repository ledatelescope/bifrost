/*
 * Copyright (c) 2019-2020, The Bifrost Authors. All rights reserved.
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

#pragma once

#include <cstdlib>      // For posix_memalign
#include <cstring>      // For memcpy, memset
#include <cstdint>

#include <arpa/inet.h>  // For ntohs

#define BF_UNPACK_FACTOR 1

#define JUMBO_FRAME_SIZE 9000

struct unaligned128_type {
    uint8_t data[16];
};
struct __attribute__((aligned(16))) aligned128_type {
    uint8_t data[16];
};
struct unaligned256_type {
        uint8_t data[32];
};
struct __attribute__((aligned(32))) aligned256_type {
	uint8_t data[32];
};
struct unaligned512_type {
        uint8_t data[64];
};
struct __attribute__((aligned(64))) aligned512_type {
	uint8_t data[64];
};


struct PacketDesc {
	uint64_t       seq;
	int            nsrc;
	int            src;
	int            nchan;
	int            chan0;
	uint32_t       sync;
	uint64_t       time_tag;
	int            tuning;
	int            tuning1 = 0;
	uint8_t        beam;
	uint16_t       gain;
	uint32_t       decimation;
	uint8_t        valid_mode;
	int            payload_size;
	const uint8_t* payload_ptr;
};


class PacketDecoder {
protected:
	int _nsrc;
	int _src0;
private:
    virtual inline bool valid_packet(const PacketDesc* pkt) const {
        return false;
    }
public:
	PacketDecoder(int nsrc, int src0) : _nsrc(nsrc), _src0(src0) {}
	virtual inline bool operator()(const uint8_t* pkt_ptr,
 	                               int            pkt_size,
	                               PacketDesc*    pkt) const {
		return false;
	}
};


class PacketProcessor {
public:
	virtual inline void operator()(const PacketDesc* pkt,
	                               uint64_t          seq0,
	                               uint64_t          nseq_per_obuf,
	                               int               nbuf,
	                               uint8_t*          obufs[],
	                               size_t            ngood_bytes[],
	                               size_t*           src_ngood_bytes[]) {}
	virtual inline void blank_out_source(uint8_t* data,
	                                     int      src,
	                                     int      nsrc,
	                                     int      nchan,
	                                     int      nseq) {}
};


class PacketHeaderFiller {
public:
    virtual inline int get_size() { return 0; }
    virtual inline void operator()(const PacketDesc* hdr_base,
                                   BFoffset          framecount,
                                   char*             hdr) {}
};
