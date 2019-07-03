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

#if BF_HWLOC_ENABLED
int HardwareLocality::bind_memory_to_core(int core) {
    int core_depth = hwloc_get_type_or_below_depth(_topo, HWLOC_OBJ_CORE);
    int ncore      = hwloc_get_nbobjs_by_depth(_topo, core_depth);
    int ret = 0;
    if( 0 <= core && core < ncore ) {
        hwloc_obj_t    obj    = hwloc_get_obj_by_depth(_topo, core_depth, core);
        hwloc_cpuset_t cpuset = hwloc_bitmap_dup(obj->allowed_cpuset);
        hwloc_bitmap_singlify(cpuset); // Avoid hyper-threads
        hwloc_membind_policy_t policy = HWLOC_MEMBIND_BIND;
        hwloc_membind_flags_t  flags  = HWLOC_MEMBIND_THREAD;
        ret = hwloc_set_membind(_topo, cpuset, policy, flags);
        hwloc_bitmap_free(cpuset);
    }
    return ret;
}
#endif // BF_HWLOC_ENABLED

BFstatus BFpacketwriter_impl::send(BFheaderinfo   desc,
                                   BFoffset       seq,
                                   BFoffset       seq_increment,
                                   BFoffset       seq_stride,
                                   BFoffset       src,
                                   BFoffset       src_increment,
                                   BFoffset       src_stride,
                                   BFarray const* in) {
    BF_ASSERT(desc,          BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(in,            BF_STATUS_INVALID_POINTER);
    BF_ASSERT(in->ndim == 1, BF_STATUS_INVALID_SHAPE);
    
    PacketDesc* hdr_base = desc->get_description();
    
    int hdr_size = _filler->get_size();
    int data_size = (BF_DTYPE_NBIT(in->dtype)/8) * _nsamples;
    int npackets = in->shape[0] / _nsamples;
    
    char* hdrs;
    hdrs = (char*) malloc(npackets*hdr_size*sizeof(char));
    for(int i=0; i<npackets; i++) {
        hdr_base->seq = seq + i*seq_increment/seq_stride;
        hdr_base->src = src + i*src_increment/src_stride;
        (*_filler)(hdr_base, hdrs+hdr_size*i);
    }
    
    _writer->send(hdrs, hdr_size, (char*) in->data, data_size, npackets);
    
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

BFstatus bfHeaderInfoSetGain(BFheaderinfo obj,
                             uint16_t     gain) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_gain(gain);
    return BF_STATUS_SUCCESS;
}

BFstatus bfHeaderInfoSetDecimation(BFheaderinfo obj,
                                   uint16_t     decimation) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_decimation(decimation);
    return BF_STATUS_SUCCESS;
}

BFstatus bfPacketWriterDestroy(BFpacketwriter obj) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    delete obj;
    return BF_STATUS_SUCCESS;
}

BFstatus bfPacketWriterSend(BFpacketwriter obj,
                            BFheaderinfo   desc,
                            BFoffset       seq,
                            BFoffset       seq_increment,
                            BFoffset       seq_stride,
                            BFoffset       src,
                            BFoffset       src_increment,
                            BFoffset       src_stride,
                            BFarray const* in) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    return obj->send(desc, seq, seq_increment, seq_stride, src, src_increment, src_stride, in);
}
