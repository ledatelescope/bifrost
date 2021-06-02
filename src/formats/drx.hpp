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

#define DRX_FRAME_SIZE 4128

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

class DRXDecoder : virtual public PacketDecoder {
    inline bool valid_packet(const PacketDesc* pkt) const {
	    return (pkt->sync       == 0x5CDEC0DE &&
	            pkt->src        >= 0 &&
        	    pkt->src        <  _nsrc &&
	            pkt->time_tag   >= 0 &&
	            (((_nsrc == 4) && 
		      (pkt->tuning     >  0 &&
		       pkt->tuning1    >  0)) ||
		     (_nsrc == 2 && 
		      (pkt->tuning     >  0 ||
		       pkt->tuning1    >  0))) &&
	            pkt->valid_mode == 0);
    }
public:
    DRXDecoder(int nsrc, int src0) : PacketDecoder(nsrc, src0) {}
    inline bool operator()(const uint8_t* pkt_ptr,
	                       int            pkt_size,
	                       PacketDesc*    pkt) const {
	    if( pkt_size != DRX_FRAME_SIZE ) {
		    return false;
	    }
	    const drx_hdr_type* pkt_hdr  = (drx_hdr_type*)pkt_ptr;
	    const uint8_t*      pkt_pld  = pkt_ptr  + sizeof(drx_hdr_type);
	    int                 pld_size = pkt_size - sizeof(drx_hdr_type);
	    int pkt_id        = pkt_hdr->frame_count_word & 0xFF;
	    pkt->beam         = (pkt_id & 0x7) - 1;
	    int pkt_tune      = ((pkt_id >> 3) & 0x7) - 1;
	    int pkt_pol       = ((pkt_id >> 7) & 0x1);
	    pkt_id            = (pkt_tune << 1) | pkt_pol;
	    pkt->sync         = pkt_hdr->sync_word;
	    pkt->time_tag     = be64toh(pkt_hdr->time_tag) - be16toh(pkt_hdr->time_offset);
	    pkt->seq          = pkt->time_tag / be16toh(pkt_hdr->decimation) / 4096;
	    pkt->nsrc         = _nsrc;
	    pkt->src          = pkt_id - _src0;
	    if( pkt->src / 2 == 0 ) {
	        pkt->tuning       = be32toh(pkt_hdr->tuning_word);
	    } else {
	        pkt->tuning1      = be32toh(pkt_hdr->tuning_word);
	    }
	    pkt->decimation   = be16toh(pkt_hdr->decimation);
	    pkt->valid_mode   = ((pkt_id >> 6) & 0x1);
	    pkt->payload_size = pld_size;
	    pkt->payload_ptr  = pkt_pld;
	    return this->valid_packet(pkt);
    }
};


class DRXProcessor : virtual public PacketProcessor {
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
	    for( ; samp<4096; ++samp ) { // HACK TESTING
		    out[samp*pkt->nsrc + pkt->src] = in[samp];
	    }
    }

    inline void blank_out_source(uint8_t* data,
                                 int      src,
                                 int      nsrc,
                                 int      nchan,
                                 int      nseq) {
	    uint8_t* __restrict__ aligned_data = (uint8_t*)data;
	    for( int t=0; t<nseq; ++t ) {
		    for( int c=0; c<4096; ++c ) {
			    aligned_data[t*4096*nsrc + c*nsrc + src] = 0;
			    aligned_data[t*4096*nsrc + c*nsrc + src] = 0;
		    }
	    }
    }
};

class DRXHeaderFiller : virtual public PacketHeaderFiller {
public:
    inline int get_size() { return sizeof(drx_hdr_type); }
    inline void operator()(const PacketDesc* hdr_base,
                           BFoffset          framecount,
                           char*             hdr) {
        drx_hdr_type* header = reinterpret_cast<drx_hdr_type*>(hdr);
        memset(header, 0, sizeof(drx_hdr_type));
        
        header->sync_word        = 0x5CDEC0DE;
        // ID is stored in the lowest 8 bits; bit 2 is reserved
        header->frame_count_word = htobe32((uint32_t) (hdr_base->src & 0xBF) << 24);
        header->decimation       = htobe16((uint16_t) hdr_base->decimation);
        header->time_offset      = 0;
        header->time_tag         = htobe64(hdr_base->seq);
        header->tuning_word      = htobe32(hdr_base->tuning);
    }
};
