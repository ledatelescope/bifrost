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

#include "assert.hpp"
#include <bifrost/packet_writer.h>
#include <bifrost/udp_transmit.h>
#include <bifrost/affinity.h>
#include "proclog.hpp"
#include "packet_writer.hpp"

#include <arpa/inet.h>  // For ntohs
#include <sys/socket.h> // For recvfrom

#include <queue>
#include <memory>
#include <stdexcept>
#include <cstdlib>      // For posix_memalign
#include <cstring>      // For memcpy, memset
#include <cstdint>

#include <sys/types.h>
#include <unistd.h>
#include <fstream>

class UDPPacketSender : public PacketWriterMethod {
public:
    UDPPacketSender(int fd)
     : PacketWriterMethod(fd) {}
    ssize_t send_packets(char* hdrs, 
                         int   hdr_size,
                         char* data, 
                         int   data_size, 
                         int   npackets,
                         int   flags=0) {
        struct mmsghdr *mmsg = NULL;
        struct iovec *iovs = NULL;
        mmsg = (struct mmsghdr *) malloc(sizeof(struct mmsghdr)*npackets);
        iovs = (struct iovec *) malloc(sizeof(struct iovec)*2*npackets);
        memset(mmsg, 0, sizeof(struct mmsghdr)*npackets);
        
        for(int i=0; i<npackets; i++) {
            mmsg[i].msg_hdr.msg_iov = &iovs[i];
            mmsg[i].msg_hdr.msg_iovlen = 2;
            iovs[2*i+0].iov_base = (hdrs + i*hdr_size);
            iovs[2*i+0].iov_len = hdr_size;
            iovs[2*i+1].iov_base = (data + i*data_size);
            iovs[2*i+1].iov_len = data_size;
        }
        
        int nsent = ::sendmmsg(_fd, mmsg, npackets, flags);
        
        free(mmsg);
        free(iovs);
        
        return nsent;
    }
    inline const char* get_name() { return "udp_transmit"; }
};

BFstatus bfUdpTransmitCreate(BFpacketwriter* obj,
                             const char*     format,
                             int             fd,
                             int             core) {
    BF_ASSERT(obj, BF_STATUS_INVALID_POINTER);
    
    int nsamples = 0;
    if(std::string(format).substr(0, 8) == std::string("generic_") ) {
        nsamples = std::atoi((std::string(format).substr(8, std::string(format).length())).c_str());
    } else if( std::string(format).substr(0, 6) == std::string("chips_") ) {
        int nchan = std::atoi((std::string(format).substr(6, std::string(format).length())).c_str());
        nsamples = 32*nchan;
    } else if(std::string(format).substr(0, 4) == std::string("cor_") ) {
        int nchan = std::atoi((std::string(format).substr(4, std::string(format).length())).c_str());
        nsamples = (8*4*nchan);
    } else if( format == std::string("tbn") ) {
        nsamples = 512;
    } else if( format == std::string("drx") ) {
        nsamples = 4096;
    } else if( format == std::string("tbf") ) {
        nsamples = 6144;
    }
    
    UDPPacketSender* method = new UDPPacketSender(fd);
    PacketWriterThread* writer = new PacketWriterThread(method, core);
    
    if( std::string(format).substr(0, 8) == std::string("generic_") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_generic_impl(writer, nsamples),
                           *obj = 0);
    } else if( std::string(format).substr(0, 6) == std::string("chips_") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_chips_impl(writer, nsamples),
                           *obj = 0);
    } else if( std::string(format).substr(0, 4) == std::string("cor_") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_cor_impl(writer, nsamples),
                           *obj = 0);
    } else if( format == std::string("tbn") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_tbn_impl(writer, nsamples),
                           *obj = 0);
    } else if( format == std::string("drx") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_drx_impl(writer, nsamples),
                           *obj = 0);
    } else if( format == std::string("tbf") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_tbf_impl(writer, nsamples),
                           *obj = 0);
    } else {
        return BF_STATUS_UNSUPPORTED;
    }
}
