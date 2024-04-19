/*
 * Copyright (c) 2019-2023, The Bifrost Authors. All rights reserved.
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

// All entries are network (i.e. big) endian
struct __attribute__((packed)) snap2_hdr_type {
        uint64_t  seq;       // Spectra counter == packet counter
        uint32_t  sync_time; // UNIX sync time
        uint16_t  npol;      // Number of pols in this packet
        uint16_t  npol_tot;  // Number of pols total
        uint16_t  nchan;     // Number of channels in this packet
        uint16_t  nchan_tot;     // Number of channels total (for this pipeline)
        uint32_t  chan_block_id; // ID of this block of chans
        uint32_t  chan0;     // First channel in this packet 
        uint32_t  pol0;      // First pol in this packet 
};

/*
 * The PacketDecoder's job is to unpack
 * a packet into a standard PacketDesc
 * format, and verify that a packet
 * is valid.
 */

#define BF_SNAP2_DEBUG 0

class SNAP2Decoder : virtual public PacketDecoder {
protected:
    inline bool valid_packet(const PacketDesc* pkt) const {
//#if BF_SNAP2_DEBUG
//        cout << "seq: "<< pkt->seq << endl;
//        cout << "src: "<< pkt->src << endl;
//        cout << "nsrc: "<< pkt->nsrc << endl;
//        cout << "nchan: "<< pkt->nchan << endl;
//        cout << "chan0: "<< pkt->chan0 << endl;
//#endif
        return ( 
                 pkt->seq >= 0
                 && pkt->src >= 0
                 && pkt->src < _nsrc
                 && pkt->nsrc == _nsrc
                 && pkt->chan0 >= 0
               );
    }
public:
    SNAP2Decoder(int nsrc, int src0) : PacketDecoder(nsrc, src0) {}
    inline bool operator()(const uint8_t* pkt_ptr,
                           int            pkt_size,
                           PacketDesc*    pkt) const {
        if( pkt_size < (int)sizeof(snap2_hdr_type) ) {
            return false;
        }
        const snap2_hdr_type* pkt_hdr  = (snap2_hdr_type*)pkt_ptr;
        const uint8_t*        pkt_pld  = pkt_ptr  + sizeof(snap2_hdr_type);
        int                   pld_size = pkt_size - sizeof(snap2_hdr_type);
        pkt->seq   = be64toh(pkt_hdr->seq);
        pkt->time_tag = be32toh(pkt_hdr->sync_time);
#if BF_SNAP2_DEBUG
	fprintf(stderr, "seq: %lu\t", pkt->seq);
	fprintf(stderr, "sync_time: %lu\t", pkt->time_tag);
	fprintf(stderr, "nchan: %lu\t", be16toh(pkt_hdr->nchan));
	fprintf(stderr, "npol: %lu\t", be16toh(pkt_hdr->npol));
#endif
        int npol_blocks  = (be16toh(pkt_hdr->npol_tot) / be16toh(pkt_hdr->npol));
        int nchan_blocks = (be16toh(pkt_hdr->nchan_tot) / be16toh(pkt_hdr->nchan));

        pkt->tuning = be32toh(pkt_hdr->chan0); // Abuse this so we can use chan0 to reference channel within pipeline
        pkt->nsrc = npol_blocks * nchan_blocks;// _nsrc;
        pkt->nchan  = be16toh(pkt_hdr->nchan);
        pkt->chan0  = be32toh(pkt_hdr->chan_block_id) * be16toh(pkt_hdr->nchan);
        pkt->nchan_tot  = be16toh(pkt_hdr->nchan_tot);
        pkt->npol  = be16toh(pkt_hdr->npol);
        pkt->npol_tot  = be16toh(pkt_hdr->npol_tot);
        pkt->pol0  = be32toh(pkt_hdr->pol0);
        pkt->src = (pkt->pol0 / pkt->npol) + be32toh(pkt_hdr->chan_block_id) * npol_blocks;
        pkt->payload_size = pld_size;
        pkt->payload_ptr  = pkt_pld;
#if BF_SNAP2_DEBUG
	fprintf(stderr, "nsrc: %lu\t", pkt->nsrc);
	fprintf(stderr, "src: %lu\t", pkt->src);
	fprintf(stderr, "chan0: %lu\t", pkt->chan0);
	fprintf(stderr, "chan_block_id: %lu\t", be32toh(pkt_hdr->chan_block_id));
	fprintf(stderr, "nchan_tot: %lu\t", pkt->nchan_tot);
	fprintf(stderr, "npol_tot: %lu\t", pkt->npol_tot);
	fprintf(stderr, "pol0: %lu\n", pkt->pol0);
#endif
        return this->valid_packet(pkt);
    }
};

class SNAP2Processor : virtual public PacketProcessor {
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
            int payload_size = pkt->payload_size;
        
            size_t obuf_offset = (pkt->seq-obuf_seq0)*pkt->nsrc*payload_size;
            typedef unaligned256_type itype; //256 bits = 32 pols / word
            typedef aligned256_type otype;
        
            obuf_offset *= BF_UNPACK_FACTOR;
        
            // Note: Using these SSE types allows the compiler to use SSE instructions
            //         However, they require aligned memory (otherwise segfault)
            itype const* __restrict__ in  = (itype const*)pkt->payload_ptr;
            otype*       __restrict__ out = (otype*      )&obufs[obuf_idx][obuf_offset];

            int words_per_chan_out = pkt->npol_tot >> 5;
            int pol_offset_out = pkt->pol0 >> 5;
            int pkt_chan = pkt->chan0;           // The first channel in this packet
        
            // Copy packet payload one channel at a time.
            // Packets have payload format nchans x npols x complexity.
            // Output buffer order is chans * npol_total * complexity
            // Spacing with which channel chunks are copied depends
            // on the total number of channels/pols in the system
            int c=0;
#if defined BF_AVX_ENABLED && BF_AVX_ENABLED
            __m256i *dest_p;
            __m256i vecbuf[2];
            uint64_t *in64 = (uint64_t *)in;
            dest_p = (__m256i *)(out + (words_per_chan_out * (pkt_chan)) + pol_offset_out);
#endif
            //if((pol_offset_out == 0) && (pkt_chan==0) && ((pkt->seq % 120)==0) ){
            //   fprintf(stderr, "nsrc: %d seq: %d, dest_p: %p obuf idx %d, obuf offset %lu, nseq_per_obuf %d, seq0 %d, nbuf: %d\n", pkt->nsrc, pkt->seq, dest_p, obuf_idx, obuf_offset, nseq_per_obuf, seq0, nbuf);
            //}
            for(c=0; c<pkt->nchan; c++) {
#if defined BF_AVX_ENABLED && BF_AVX_ENABLED
               vecbuf[0] = _mm256_set_epi64x(in64[3], in64[2], in64[1], in64[0]);
               vecbuf[1] = _mm256_set_epi64x(in64[7], in64[6], in64[5], in64[4]);
               _mm256_stream_si256(dest_p, vecbuf[0]);
               _mm256_stream_si256(dest_p+1,   vecbuf[1]);
               in64 += 8;
               dest_p += words_per_chan_out;
#else
               ::memcpy(&out[pkt->src + pkt->nsrc*c],
                        &in[c], sizeof(otype));
#endif
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
                           ::memset(&aligned_data[src + nsrc*(c + nchan*t)],
                                    0, sizeof(otype));
                   }
            }
    }
};

class SNAP2HeaderFiller : virtual public PacketHeaderFiller {
public:
    inline int get_size() { return sizeof(snap2_hdr_type); }
    inline void operator()(const PacketDesc* hdr_base,
                           BFoffset          framecount,
                           char*             hdr) {
        snap2_hdr_type* header = reinterpret_cast<snap2_hdr_type*>(hdr);
        memset(header, 0, sizeof(snap2_hdr_type));
        
        header->seq           = htobe64(hdr_base->seq);
        header->npol          = 2;
        header->npol_tot      = 2;
        header->nchan         = hdr_base->nchan;
        header->nchan_tot     = hdr_base->nchan * hdr_base->nsrc;
        header->chan_block_id = hdr_base->src;
        header->chan0         = htons(hdr_base->chan0);
        header->pol0          = 0;
        
    }
};
