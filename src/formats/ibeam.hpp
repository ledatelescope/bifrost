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

//#include <immintrin.h> // SSE

#pragma pack(1)
struct ibeam_hdr_type {
	uint8_t  server;   // Note: 1-based
	uint8_t  gbe;      // (AKA tuning)
	uint8_t  nchan;    // 109
	uint8_t  nbeam;    // 2
	uint8_t  nserver;  // 6
	// Note: Big endian
	uint16_t chan0;    // First chan in packet
	uint64_t seq;      // Note: 1-based
};

template<uint8_t B>
class IBeamDecoder: virtual public PacketDecoder {
    uint8_t _nbeam = B;
    
	inline bool valid_packet(const PacketDesc* pkt) const {
        return (pkt->seq   >= 0 &&
		        pkt->src   >= 0 && pkt->src < _nsrc &&
		        pkt->beam  == _nbeam &&
		        pkt->chan0 >= 0);
    }
public:
    IBeamDecoder(int nsrc, int src0) : PacketDecoder(nsrc, src0) {}
    inline bool operator()(const uint8_t* pkt_ptr,
	                       int            pkt_size,
	                       PacketDesc*    pkt) const {
	    if( pkt_size < (int)sizeof(ibeam_hdr_type) ) {
	        return false;
	    }
	    const ibeam_hdr_type* pkt_hdr  = (ibeam_hdr_type*)pkt_ptr;
	    const uint8_t*        pkt_pld  = pkt_ptr  + sizeof(ibeam_hdr_type);
	    int                   pld_size = pkt_size - sizeof(ibeam_hdr_type);
	    pkt->seq   = be64toh(pkt_hdr->seq)  - 1;
	    //pkt->nsrc  =         pkt_hdr->nserver;
	    pkt->nsrc  =         _nsrc;
	    pkt->src   =        (pkt_hdr->server - 1) - _src0;
	    pkt->beam  =         pkt_hdr->nbeam;
	    pkt->nchan =         pkt_hdr->nchan;
	    pkt->chan0 =   ntohs(pkt_hdr->chan0) - pkt->nchan * pkt->src;
	    pkt->payload_size = pld_size;
	    pkt->payload_ptr  = pkt_pld;
	    return this->valid_packet(pkt);
    }
};

template<uint8_t B>
class IBeamProcessor : virtual public PacketProcessor {
    uint8_t _nbeam = B;
    
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
	    size_t nbyte = pkt->payload_size * BF_UNPACK_FACTOR;
	    ngood_bytes[obuf_idx]               += nbyte;
	    src_ngood_bytes[obuf_idx][pkt->src] += nbyte;
	    // **CHANGED RECENTLY
	    int payload_size = pkt->payload_size;
	
	    size_t obuf_offset = (pkt->seq-obuf_seq0)*pkt->nsrc*payload_size;
	    typedef unaligned128_type itype;  // cf32 * 1 beam * 2 pol
	    typedef aligned128_type otype;
	
	    obuf_offset *= BF_UNPACK_FACTOR;
	
	    // Note: Using these SSE types allows the compiler to use SSE instructions
	    //         However, they require aligned memory (otherwise segfault)
	    itype const* __restrict__ in  = (itype const*)pkt->payload_ptr;
	    otype*       __restrict__ out = (otype*      )&obufs[obuf_idx][obuf_offset];
	
	    int chan, beam;
	    for(chan=0; chan<pkt->nchan; ++chan ) {
		    for(beam=0; beam<_nbeam; ++beam) {
			    ::memcpy(&out[pkt->src*pkt->nchan*_nbeam + chan*_nbeam + beam], 
			 	     &in[chan*_nbeam + beam], sizeof(otype));
                	    //out[pkt->src*pkt->nchan*_nbeam + chan*_nbeam + beam] = in[chan*_nbeam + beam];
		    }
	    }
    }

    inline void blank_out_source(uint8_t* data,
	                             int      src,
	                             int      nsrc,
	                             int      nchan,
	                             int      nseq) {
	    typedef aligned128_type otype;
	    otype* __restrict__ aligned_data = (otype*)data;
	    for( int t=0; t<nseq; ++t ) {
		for( int c=0; c<nchan; ++c ) {
                	for( int b=0; b<_nbeam; ++b ) {
                    		::memset(&aligned_data[t*nsrc*nchan*_nbeam + src*nchan*_nbeam + c*_nbeam + b],
			                 0, sizeof(otype));
                	}
		}
	    }
    }
};

template<uint8_t B>
class IBeamHeaderFiller : virtual public PacketHeaderFiller {
    uint8_t _nbeam = B;
    
public:
    inline int get_size() { return sizeof(ibeam_hdr_type); }
    inline void operator()(const PacketDesc* hdr_base,
                           BFoffset          framecount,
                           char*             hdr) {
        ibeam_hdr_type* header = reinterpret_cast<ibeam_hdr_type*>(hdr);
        memset(header, 0, sizeof(ibeam_hdr_type));
        
        header->server   = hdr_base->src + 1;
        header->gbe      = hdr_base->tuning;
        header->nchan    = hdr_base->nchan;
        header->nbeam    = _nbeam;
        header->nserver  = hdr_base->nsrc;
        header->chan0    = htons(hdr_base->chan0);
        header->seq      = htobe64(hdr_base->seq);
    }
};
