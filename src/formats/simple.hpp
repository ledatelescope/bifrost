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

struct __attribute__((packed)) simple_hdr_type {
	uint32_t seq; 
};

class SIMPLEDecoder : virtual public PacketDecoder {
	inline bool valid_packet(const PacketDesc* pkt) const {
        return (pkt->seq   >= 0 );
    }
public:
    SIMPLEDecoder(int nsrc, int src0) : PacketDecoder(nsrc, src0) {}
    inline bool operator()(const uint8_t* pkt_ptr,
	                       int            pkt_size,
	                       PacketDesc*    pkt) const {
        if( pkt_size < (int)sizeof(simple_hdr_type) ) {
	        return false;
	    }
	    const simple_hdr_type* pkt_hdr  = (simple_hdr_type*)pkt_ptr;
	    const uint8_t*        pkt_pld  = pkt_ptr  + sizeof(simple_hdr_type);
	    int                   pld_size = pkt_size - sizeof(simple_hdr_type);
	    pkt->seq =   be32toh(pkt_hdr->seq);
            pkt->nsrc = 1;
            pkt->src = 0;
	    pkt->payload_size = pld_size;
	    pkt->payload_ptr  = pkt_pld;
	    pkt->nchan =         1;
	    pkt->chan0 =   0;
	    return this->valid_packet(pkt);
    }
};

class SIMPLEProcessor : virtual public PacketProcessor {
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
	    src_ngood_bytes[obuf_idx][0] += nbyte;
	    // **CHANGED RECENTLY
	    int payload_size = pkt->payload_size;//pkt->nchan*(PKT_NINPUT*2*PKT_NBIT/8);
	
	    size_t obuf_offset = (pkt->seq-obuf_seq0)*pkt->nsrc*payload_size;
	    typedef unaligned256_type itype;
	    typedef aligned256_type otype;
	
	    obuf_offset *= BF_UNPACK_FACTOR;
	
	    // Note: Using these SSE types allows the compiler to use SSE instructions
	    //         However, they require aligned memory (otherwise segfault)
	    itype const* __restrict__ in  = (itype const*)pkt->payload_ptr;
	    otype*       __restrict__ out = (otype*      )&obufs[obuf_idx][obuf_offset];
	
	    int chan = 0;
			//cout << pkt->src << ", " << pkt->nsrc << endl;
		  //cout << pkt->nchan << endl;
			for( ; chan<pkt->nchan; ++chan ) {
/* #if defined BF_AVX_ENABLED && BF_AVX_ENABLED
           _mm256_store_si256(reinterpret_cast<__m256i*>(&out[pkt->src + pkt->nsrc*chan]),
					                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&in[chan])));
#else
#if defined BF_SSE_ENABLED && BF_SSE_ENABLED
           const unaligned128_type* dsrc = (const unaligned128_type*) &in[chan];
           aligned128_type* ddst = (aligned128_type*) &out[pkt->src + pkt->nsrc*chan];
           
           _mm_store_si128(reinterpret_cast<__m128i*>(ddst),
					                 _mm_loadu_si128(reinterpret_cast<const __m128i*>(dsrc)));
           _mm_store_si128(reinterpret_cast<__m128i*>(ddst+1),
					                 _mm_loadu_si128(reinterpret_cast<const __m128i*>(dsrc+1)));
#else*/
						::memcpy(&out[pkt->src + pkt->nsrc*chan],
						      	 &in[chan], 256*sizeof(otype));
// #endif
// #endif
      }
    }

    inline void blank_out_source(uint8_t* data,
	                             int      src,
	                             int      nsrc,
	                             int      nchan,
	                             int      nseq) {
	    typedef aligned256_type otype;
	    otype* __restrict__ aligned_data = (otype*)data;
	    for( int t=0; t<nseq; ++t ) {
		    for( int c=0; c<nchan; ++c ) {
/* #if defined BF_AVX_ENABLED && BF_AVX_ENABLED
			    _mm256_store_si256(reinterpret_cast<__m256i*>(&aligned_data[src + nsrc*(c + nchan*t)]),
					                   _mm256_setzero_si256());
#else
#if defined BF_SSE_ENABLED && BF_SSE_ENABLED
			    aligned128_type* ddst = (aligned128_type*) &aligned_data[src + nsrc*(c + nchan*t)];
					
					_mm_store_si128(reinterpret_cast<__m128i*>(ddst),
					                _mm_setzero_si128());
					_mm_store_si128(reinterpret_cast<__m128i*>(ddst+1),
					                _mm_setzero_si128());
#else*/
			    ::memset(&aligned_data[src + nsrc*(c + nchan*t)],
			             0, sizeof(otype));
// #endif
// #endif
		    }
	    }
    }
};

class SIMPLEHeaderFiller : virtual public PacketHeaderFiller {
public:
    inline int get_size() { return sizeof(simple_hdr_type); }
    inline void operator()(const PacketDesc* hdr_base,
                           BFoffset          framecount,
                           char*             hdr) {
        simple_hdr_type* header = reinterpret_cast<simple_hdr_type*>(hdr);
        memset(header, 0, sizeof(simple_hdr_type));
        
        header->seq      = htobe32(hdr_base->seq);
    }
};
