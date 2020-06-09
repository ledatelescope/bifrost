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

#pragma once

#include "base.hpp"

#define TBN_FRAME_SIZE 1048

#pragma pack(1)
struct tbn_hdr_type {
	uint32_t sync_word;
	uint32_t frame_count_word;
	uint32_t tuning_word;
	uint16_t tbn_id;
	uint16_t gain;
	uint64_t time_tag;
};

class TBNCache {
    uint64_t _timestamp_last = 0;
    int      _pid_last = -1000;
    uint16_t _decimation = 1;
    bool     _active = false;
public:
    TBNCache() {}
    inline uint16_t get_decimation() { return _decimation; }
    inline bool update(int      packet_id,
                       uint64_t timestamp) {
        if( !_active ) {
            if( packet_id == _pid_last ) {
                _decimation = ((timestamp - _timestamp_last) / 512);
                _active = true;
            } else if( _pid_last == -1000 ) {
                _pid_last = packet_id;
                _timestamp_last = timestamp;
            }
        }
        return _active;
    }
};

class TBNDecoder : virtual public PacketDecoder {
    TBNCache* _cache;
    
    inline bool valid_packet(const PacketDesc* pkt) const {
	    return (pkt->sync       == 0x5CDEC0DE &&
	            pkt->src        >= 0 && 
		    pkt->src        <  _nsrc &&
	            pkt->time_tag   >= 0 &&
	            pkt->tuning     >= 0 && 
		    pkt->valid_mode == 0);
    }
public:
    TBNDecoder(int nsrc, int src0)
      : PacketDecoder(nsrc, src0) {
        _cache = new TBNCache();
    }
    inline bool operator()(const uint8_t* pkt_ptr,
	                       int            pkt_size,
	                       PacketDesc*    pkt) const {
	    if( pkt_size != TBN_FRAME_SIZE ) {
		    return false;
	    }
	    const tbn_hdr_type* pkt_hdr  = (tbn_hdr_type*)pkt_ptr;
	    const uint8_t*      pkt_pld  = pkt_ptr  + sizeof(tbn_hdr_type);
	    int                 pld_size = pkt_size - sizeof(tbn_hdr_type);
	    pkt->sync         = pkt_hdr->sync_word;
	    pkt->time_tag     = be64toh(pkt_hdr->time_tag);
	    pkt->nsrc         = _nsrc;
	    pkt->src          = (be16toh(pkt_hdr->tbn_id) & 1023) - 1 - _src0;
	    pkt->tuning       = be32toh(pkt_hdr->tuning_word);
	    pkt->valid_mode   = (be16toh(pkt_hdr->tbn_id) >> 15) & 1;
	    pkt->gain         = (be16toh(pkt_hdr->gain));
	    pkt->payload_size = pld_size;
	    pkt->payload_ptr  = pkt_pld;
	    bool valid        = this->valid_packet(pkt);
	    if( valid ) {
		    valid &= _cache->update(pkt->src, pkt->time_tag);
		    pkt->decimation = _cache->get_decimation();
		    pkt->seq        = pkt->time_tag / pkt->decimation / 512;
	    }
	    return valid;
    }
};

class TBNProcessor : virtual public PacketProcessor {
public:
    inline void operator()(const PacketDesc* pkt,
	                       uint64_t          seq0,
	                       uint64_t          nseq_per_obuf,
	                       int               nbuf,
	                       uint8_t*          obufs[],
	                       size_t            ngood_bytes[],
	                       size_t*           src_ngood_bytes[]) {
	    int    obuf_idx = ((pkt->seq - seq0 >= 1*nseq_per_obuf) +
	                       (pkt->seq - seq0 >= 2*nseq_per_obuf));
	    size_t obuf_seq0 = seq0 + obuf_idx*nseq_per_obuf;
	    size_t nbyte = pkt->payload_size;
	    ngood_bytes[obuf_idx]               += nbyte;
	    src_ngood_bytes[obuf_idx][pkt->src] += nbyte;
	    int payload_size = pkt->payload_size;
	
	    size_t obuf_offset = (pkt->seq-obuf_seq0)*pkt->nsrc*payload_size;
	
	    // Note: Using these SSE types allows the compiler to use SSE instructions
	    //         However, they require aligned memory (otherwise segfault)
	    uint8_t const* __restrict__ in  = (uint8_t const*)pkt->payload_ptr;
	    uint8_t*       __restrict__ out = (uint8_t*      )&obufs[obuf_idx][obuf_offset];
	
	    int samp = 0;
	    for( ; samp<512; ++samp ) { // HACK TESTING
		    out[samp*pkt->nsrc*2 + pkt->src*2 + 0] = in[2*samp+0];
		    out[samp*pkt->nsrc*2 + pkt->src*2 + 1] = in[2*samp+1];
	    }
    }
	
    inline void blank_out_source(uint8_t* data,
	                             int      src,
	                             int      nsrc,
	                             int      nchan,
	                             int      nseq) {
	    uint8_t* __restrict__ aligned_data = (uint8_t*)data;
	    for( int t=0; t<nseq; ++t ) {
		    for( int c=0; c<512; ++c ) {
			    aligned_data[t*512*nsrc*2 + c*nsrc*2 + src*2 + 0] = 0;
			    aligned_data[t*512*nsrc*2 + c*nsrc*2 + src*2 + 1] = 0;
		    }
	    }
    }
};

class TBNHeaderFiller : virtual public PacketHeaderFiller {
public:
    inline int get_size() { return sizeof(tbn_hdr_type); }
    inline void operator()(const PacketDesc* hdr_base,
                           BFoffset          framecount,
                           char*             hdr) {
        tbn_hdr_type* header = reinterpret_cast<tbn_hdr_type*>(hdr);
        memset(header, 0, sizeof(tbn_hdr_type));
        
        header->sync_word        = 0x5CDEC0DE;
        // Bits 9-32 are the frame count; bits 1-8 are zero
        header->frame_count_word = htobe32((framecount & 0xFFFFFF));
        header->tuning_word      = htobe32(hdr_base->tuning);
        // ID is the upper 14 bits; bit 2 is reserved; bit 1 is the TBW flag
        header->tbn_id           = htons((hdr_base->src + 1) & 0x3FFF);
        header->gain             = htons(hdr_base->gain);
        header->time_tag         = htobe64(hdr_base->seq);
    }
};
