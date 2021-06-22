/*
 * Copyright (c) 2020, The Bifrost Authors. All rights reserved.
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
struct pbeam_hdr_type {
	uint8_t  server;   // Note: 1-based
	uint8_t  beam;     // Note: 1-based
	uint8_t  gbe;      // (AKA tuning)
	uint8_t  nchan;    // 109
	uint8_t  nbeam;    // 2
	uint8_t  nserver;  // 6
	// Note: Big endian
	uint16_t navg;     // Number of raw spectra averaged
	uint16_t chan0;    // First chan in packet
	uint64_t seq;      // Note: 1-based
};

class PBeamDecoder: virtual public PacketDecoder {
   inline bool valid_packet(const PacketDesc* pkt) const {
        return (pkt->seq   >= 0 &&
		        		pkt->src   >= 0 && pkt->src < _nsrc &&
								pkt->chan0 >= 0);
    }
public:
    PBeamDecoder(int nsrc, int src0) : PacketDecoder(nsrc, src0) {}
    inline bool operator()(const uint8_t* pkt_ptr,
	                         int            pkt_size,
	                         PacketDesc*    pkt) const {
	    if( pkt_size < (int)sizeof(pbeam_hdr_type) ) {
	        return false;
	    }
	    const pbeam_hdr_type* pkt_hdr  = (pbeam_hdr_type*)pkt_ptr;
	    const uint8_t*        pkt_pld  = pkt_ptr  + sizeof(pbeam_hdr_type);
	    int                   pld_size = pkt_size - sizeof(pbeam_hdr_type);
	    pkt->decimation =    be16toh(pkt_hdr->navg);
	    pkt->time_tag   =    be64toh(pkt_hdr->seq);
	    pkt->seq   =         (pkt->time_tag - 1) / pkt->decimation;
	    //pkt->nsrc  =         pkt_hdr->nserver;
	    pkt->nsrc  =         _nsrc;
	    pkt->src   =         (pkt_hdr->beam - _src0) * pkt_hdr->nserver + (pkt_hdr->server - 1);
			pkt->beam  =         pkt_hdr->nbeam;
	    pkt->nchan =         pkt_hdr->nchan;
	    pkt->chan0 =   ntohs(pkt_hdr->chan0) - pkt->nchan * pkt->src;
	    pkt->payload_size = pld_size;
	    pkt->payload_ptr  = pkt_pld;
	    return this->valid_packet(pkt);
    }
};

class PBeamProcessor : virtual public PacketProcessor {
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
	    typedef unaligned128_type itype;  // f32 * 1 beam * 4 pol (XX, YY, R(XY), I(XY))
	    typedef aligned128_type otype;
	
	    obuf_offset *= BF_UNPACK_FACTOR;
	
	    // Note: Using these SSE types allows the compiler to use SSE instructions
	    //         However, they require aligned memory (otherwise segfault)
	    itype const* __restrict__ in  = (itype const*)pkt->payload_ptr;
	    otype*       __restrict__ out = (otype*      )&obufs[obuf_idx][obuf_offset];
			
			// Convenience
	    int beam_server = pkt->src;
	    int nchan = pkt->nchan;
	    
	    int chan;
	    for(chan=0; chan<nchan; ++chan ) {
		    ::memcpy(&out[beam_server*nchan + chan], &in[chan], sizeof(otype));
	    }
    }

    inline void blank_out_source(uint8_t* data,
	                               int      src,	// beam_server
	                               int      nsrc,	// nbeam_server
	                               int      nchan,
	                               int      nseq) {
	    typedef aligned128_type otype;
	    otype* __restrict__ aligned_data = (otype*)data;
			for( int t=0; t<nseq; ++t ) {
				::memset(&aligned_data[t*nsrc*nchan + src*nchan],
			           0, nchan*sizeof(otype));
      }
    }
};

template<uint8_t B>
class PBeamHeaderFiller : virtual public PacketHeaderFiller {
		uint8_t _nbeam = B;
	
public:
    inline int get_size() { return sizeof(pbeam_hdr_type); }
    inline void operator()(const PacketDesc* hdr_base,
                           BFoffset          framecount,
                           char*             hdr) {
        pbeam_hdr_type* header = reinterpret_cast<pbeam_hdr_type*>(hdr);
        memset(header, 0, sizeof(pbeam_hdr_type));
        
				int nserver = hdr_base->nsrc / _nbeam;
				
        header->server   = (hdr_base->src % nserver) + 1;	// Modulo?
				header->beam     = (hdr_base->src / nserver) + 1;
        header->gbe      = hdr_base->tuning;
        header->nchan    = hdr_base->nchan;
        header->nbeam    = _nbeam;
        header->nserver  = nserver;
        header->navg     = htons((uint16_t) hdr_base->decimation);
        header->chan0    = htons((uint16_t) hdr_base->chan0);
        header->seq      = htobe64(hdr_base->seq);
    }
};
