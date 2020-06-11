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
 
#include "packet_writer.hpp"

#define BF_JAYCE_DEBUG 0

#if BF_JAYCE_DEBUG
#define BF_PRINTD(stmt) \
    std::cout << stmt << std::endl
#else // not BF_JAYCE_DEBUG
#define BF_PRINTD(stmt)
#endif

BFstatus BFpacketwriter_impl::send(BFheaderinfo   info,
                                   BFoffset       seq,
                                   BFoffset       seq_increment,
                                   BFoffset       src,
                                   BFoffset       src_increment,
                                   BFarray const* in) {
    BF_ASSERT(info,          BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(in,            BF_STATUS_INVALID_POINTER);
    BF_ASSERT(in->dtype == _dtype,       BF_STATUS_INVALID_DTYPE);
    BF_ASSERT(in->ndim == 3,             BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(in->shape[2] == _nsamples, BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(is_contiguous(in),  BF_STATUS_UNSUPPORTED_STRIDE);
    BF_ASSERT(space_accessible_from(in->space, BF_SPACE_SYSTEM),
              BF_STATUS_UNSUPPORTED_SPACE);
    
    PacketDesc* hdr_base = info->get_description();
    
    int i, j;
    int hdr_size = _filler->get_size();
    int data_size = (BF_DTYPE_NBIT(in->dtype)/8) * _nsamples;
    int npackets = in->shape[0]*in->shape[1];
    
    char* hdrs;
    hdrs = (char*) malloc(npackets*hdr_size*sizeof(char));
    for(i=0; i<in->shape[0]; i++) {
        hdr_base->seq = seq + i*seq_increment;
        for(j=0; j<in->shape[1]; j++) {
            hdr_base->src = src + j*src_increment;
            (*_filler)(hdr_base, _framecount, hdrs+hdr_size*(i*in->shape[1]+j));
        }
        _framecount++;
    }
    
    _writer->send(hdrs, hdr_size, (char*) in->data, data_size, npackets);
    this->update_stats_log();
    
    free(hdrs);
    return BF_STATUS_SUCCESS;
}

BFstatus bfHeaderInfoCreate(BFheaderinfo* obj) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    BF_TRY_RETURN_ELSE(*obj = new BFheaderinfo_impl(),
                       *obj = 0);
}

BFstatus bfHeaderInfoDestroy(BFheaderinfo obj) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    delete obj;
    return BF_STATUS_SUCCESS;
}

BFstatus bfHeaderInfoSetNSrc(BFheaderinfo obj,
                             int          nsrc) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_nsrc(nsrc);
    return BF_STATUS_SUCCESS;
}

BFstatus bfHeaderInfoSetNChan(BFheaderinfo obj,
                              int          nchan) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_nchan(nchan);
    return BF_STATUS_SUCCESS;
}

BFstatus bfHeaderInfoSetChan0(BFheaderinfo obj,
                              int          chan0) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_chan0(chan0);
    return BF_STATUS_SUCCESS;
}

BFstatus bfHeaderInfoSetTuning(BFheaderinfo obj,
                               int          tuning) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_tuning(tuning);
    return BF_STATUS_SUCCESS;
}

BFstatus bfHeaderInfoSetGain(BFheaderinfo       obj,
                             unsigned short int gain) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_gain(gain);
    return BF_STATUS_SUCCESS;
}

BFstatus bfHeaderInfoSetDecimation(BFheaderinfo       obj,
                                   unsigned short int decimation) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_decimation(decimation);
    return BF_STATUS_SUCCESS;
}

BFstatus bfDiskWriterCreate(BFpacketwriter* obj,
                            const char*     format,
                            int             fd,
                            int             core) {
    return BFpacketwriter_create(obj, format, fd, core, BF_IO_DISK);
}

BFstatus bfUdpTransmitCreate(BFpacketwriter* obj,
                             const char*     format,
                             int             fd,
                             int             core) {
    return BFpacketwriter_create(obj, format, fd, core, BF_IO_UDP);
}

BFstatus bfPacketWriterDestroy(BFpacketwriter obj) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    delete obj;
    return BF_STATUS_SUCCESS;
}

BFstatus bfPacketWriterResetCounter(BFpacketwriter obj) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    BF_TRY_RETURN(obj->reset_counter());
}

BFstatus bfPacketWriterSend(BFpacketwriter obj,
                            BFheaderinfo   info,
                            BFoffset       seq,
                            BFoffset       seq_increment,
                            BFoffset       src,
                            BFoffset       src_increment,
                            BFarray const* in) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    return obj->send(info, seq, seq_increment, src, src_increment, in);
}
