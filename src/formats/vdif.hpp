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

#pragma pack(1)
struct vdif_hdr_type {
    struct word_1_ {
        uint32_t seconds_from_epoch:30;
        uint8_t  is_legacy:1;
        uint8_t  is_invalid:1;
    } word_1;
    struct word_2_ {
        uint32_t frame_in_second:24;
        uint16_t ref_epoch:6;
        uint8_t  unassigned:2;
    } word_2;
    struct word_3_ {
        uint32_t frame_length:24;
        uint32_t log2_nchan:5;
        uint8_t  version:3;
    } word_3;
    struct word_4_ {
        uint16_t station_id:16;
        uint16_t thread_id:10;
        uint8_t  bits_per_sample_minus_one:5;
        uint8_t  is_complex:1;
    } word_4;
};

#pragma pack(1)
struct vdif_ext_type {
    uint32_t extended_word_1;
    uint32_t extended_word_2;
    uint32_t extended_word_3;
    uint32_t extended_word_4;
};

class VDIFCache {
    BFoffset _frame_last = 0;
    uint32_t _frames_per_second = 0;
    uint32_t _sample_rate = 0;
    bool     _active = false;
public:
    VDIFCache() {}
    inline uint32_t get_frames_per_second() { return _frames_per_second; }
    inline uint32_t get_sample_rate() { return _sample_rate; }
    inline bool update(uint32_t seconds_from_epoch,
                       uint32_t frame_in_second,
                       uint32_t samples_per_frame) {
        if( !_active ) {
            if( frame_in_second < _frame_last ) {
                _frames_per_second = _frame_last + 1;
                _sample_rate = _frames_per_second * samples_per_frame;
                _active = true;
            }
            _frame_last = frame_in_second;
        }
        return _active;
    }
};

class VDIFDecoder : virtual public PacketDecoder {
    VDIFCache* _cache;
    
    inline bool valid_packet(const PacketDesc* pkt) const {
        return (pkt->src           >= 0 &&
                pkt->src           <  _nsrc &&
                pkt->valid_mode    == 0);
    }
public:
    VDIFDecoder(int nsrc, int src0)
      : PacketDecoder(nsrc, src0) {
        _cache = new VDIFCache();
    }
    inline bool operator()(const uint8_t* pkt_ptr,
                           int            pkt_size,
                           PacketDesc*    pkt) const {
        if( pkt_size < (int)sizeof(vdif_hdr_type) ) {
            return false;
        }
        const vdif_hdr_type* pkt_hdr  = (vdif_hdr_type*)pkt_ptr;
        const uint8_t*       pkt_pld  = pkt_ptr  + sizeof(vdif_hdr_type);
        int                  pld_size = pkt_size - sizeof(vdif_hdr_type);
        if( pkt_hdr->word_1.is_legacy == 0 ) {
            // Do not try to decode the extended header entries
            pkt_pld  += sizeof(vdif_ext_type);
            pld_size -= sizeof(vdif_ext_type);
        }
        
        uint32_t nsamples = pld_size * 8 \
                            / (pkt_hdr->word_4.bits_per_sample_minus_one + 1) \
                            / ((int) 1 << pkt_hdr->word_3.log2_nchan) \
                            / (1 + pkt_hdr->word_4.is_complex);
        bool is_active = _cache->update(pkt_hdr->word_1.seconds_from_epoch,
                                        pkt_hdr->word_2.frame_in_second,
                                        nsamples);
        
        pkt->seq          = (BFoffset) pkt_hdr->word_1.seconds_from_epoch*_cache->get_frames_per_second() \
                            + (BFoffset) pkt_hdr->word_2.frame_in_second;
        pkt->time_tag     = pkt->seq*_cache->get_sample_rate();
        pkt->src          = pkt_hdr->word_4.thread_id - _src0;
        pkt->nsrc         = _nsrc;
        pkt->chan0        = (int) 1 << pkt_hdr->word_3.log2_nchan;
        pkt->nchan        = pld_size / 8;
        pkt->sync         = _cache->get_sample_rate();
        pkt->tuning       = (((int) pkt_hdr->word_2.ref_epoch) << 16) \
                            | (((int) pkt_hdr->word_4.bits_per_sample_minus_one + 1) << 8) \
                            | pkt_hdr->word_4.is_complex;
        pkt->valid_mode   = pkt_hdr->word_1.is_invalid || (not is_active);
        pkt->payload_size = pld_size;
        pkt->payload_ptr  = pkt_pld;
        /*
        if( this->valid_packet(pkt) && (pkt_hdr->word_2.frame_in_second == 1) ) {
            std::cout << "pld_size:           " << pld_size << std::endl;
            std::cout << "        :           " << pkt_hdr->word_3.frame_length*8 - 32 + 16*pkt_hdr->word_1.is_legacy<< std::endl;
            std::cout << "seconds_from_epoch: " << pkt_hdr->word_1.seconds_from_epoch << std::endl;
            std::cout << "frame_in_second:    " << pkt_hdr->word_2.frame_in_second << std::endl;
            std::cout << "frames_per_second:  " << _cache->get_frames_per_second() << std::endl;
            std::cout << "sample_rate:        " << _cache->get_sample_rate() << std::endl;
            std::cout << "src:                " << pkt->src << std::endl;
            std::cout << "is_legacy:          " << pkt_hdr->word_1.is_legacy << std::endl;
            std::cout << "valid:              " << this->valid_packet(pkt) << std::endl;
        }
        */
        return this->valid_packet(pkt);
    }
};

class VDIFProcessor : virtual public PacketProcessor {
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
        typedef uint64_t itype;      // 8-byte chunks in the file
        typedef uint64_t otype;
    
        // Note: Using these SSE types allows the compiler to use SSE instructions
        //         However, they require aligned memory (otherwise segfault)
        itype const* __restrict__ in  = (itype const*)pkt->payload_ptr;
        otype*       __restrict__ out = (otype*      )&obufs[obuf_idx][obuf_offset];
        
        // Convenience
        int src = pkt->src;
        int nchan = pkt->nchan;
        
        int chan = 0;
        for( ; chan<nchan; ++chan ) {
            out[src*nchan + chan] = in[chan];
        }
    }
    
    inline void blank_out_source(uint8_t* data,
                                 int      src,
                                 int      nsrc,
                                 int      nchan,
                                 int      nseq) {
        typedef uint64_t otype;
        otype* __restrict__ aligned_data = (otype*)data;
        for( int t=0; t<nseq; ++t ) {
            ::memset(&aligned_data[t*nsrc*nchan + src*nchan],
                     0, nchan*sizeof(otype));
        }
    }
};
