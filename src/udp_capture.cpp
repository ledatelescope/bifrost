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
#include <bifrost/packet_capture.h>
#include <bifrost/udp_capture.h>
#include <bifrost/affinity.h>
#include <bifrost/Ring.hpp>
using bifrost::ring::RingWrapper;
using bifrost::ring::RingWriter;
using bifrost::ring::WriteSpan;
using bifrost::ring::WriteSequence;
#include "proclog.hpp"
#include "formats/formats.hpp"
#include "packet_capture.hpp"

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
#include <chrono>

// TODO: The VMA API is returning unaligned buffers, which prevents use of SSE
#ifndef BF_VMA_ENABLED
#define BF_VMA_ENABLED 0
//#define BF_VMA_ENABLED 1
#endif

#if BF_VMA_ENABLED
#include <mellanox/vma_extra.h>
class VMAReceiver {
	int           _fd;
	vma_api_t*    _api;
	vma_packet_t* _pkt;
	inline void clean_cache() {
		if( _pkt ) {
			_api->free_packets(_fd, _pkt, 1);
			_pkt = 0;
		}
	}
public:
	VMAReceiver(int fd)
		: _fd(fd), _api(vma_get_api()), _pkt(0) {}
	VMAReceiver(VMAReceiver const& other)
		: _fd(other._fd), _api(other._api), _pkt(0) {}
	VMAReceiver& operator=(VMAReceiver const& other) {
		if( &other != this ) {
			this->clean_cache();
			_fd  = other._fd;
			_api = other._api;
		}
		return *this;
	}
	~VMAReceiver() { this->clean_cache(); }
	inline int recv_packet(uint8_t* buf, size_t bufsize, uint8_t** pkt_ptr, int flags=0) {
		this->clean_cache();
		int ret = _api->recvfrom_zcopy(_fd, buf, bufsize, &flags, 0, 0);
		if( ret < 0 ) {
			return ret;
		}
		if( flags & MSG_VMA_ZCOPY ) {
			_pkt = &((vma_packets_t*)buf)->pkts[0];
			*pkt_ptr = (uint8_t*)_pkt->iov[0].iov_base;
		} else {
			*pkt_ptr = buf;
		}
		return ret;
	}
	inline operator bool() const { return _api != NULL; }
};
#endif // BF_VMA_ENABLED

class UDPPacketReceiver : public DataCaptureMethod {
#if BF_VMA_ENABLED
	VMAReceiver            _vma;
#endif
public:
	UDPPacketReceiver(int fd, size_t pkt_size_max=JUMBO_FRAME_SIZE)
		: DataCaptureMethod(fd, pkt_size_max)
#if BF_VMA_ENABLED
		, _vma(fd)
#endif
	{}
	inline int recv_packet(uint8_t** pkt_ptr, int flags=0) {
#if BF_VMA_ENABLED
		if( _vma ) {
			*pkt_ptr = 0;
			return _vma.recv_packet(&_buf[0], _buf.size(), pkt_ptr, flags);
		} else {
#endif
			*pkt_ptr = &_buf[0];
			return ::recvfrom(_fd, &_buf[0], _buf.size(), flags, 0, 0);
#if BF_VMA_ENABLED
		}
#endif
	}
	inline const char* get_name() { return "udp_capture"; }
};

BFstatus bfUdpCaptureCreate(BFdatacapture* obj,
                            const char*    format,
                            int            fd,
                            BFring         ring,
                            BFsize         nsrc,
                            BFsize         src0,
                            BFsize         max_payload_size,
                            BFsize         buffer_ntime,
                            BFsize         slot_ntime,
                            BFdatacapture_callback sequence_callback,
                            int            core) {
	BF_ASSERT(obj, BF_STATUS_INVALID_POINTER);
	
	UDPPacketReceiver* method = new UDPPacketReceiver(fd, max_payload_size);
	DataCaptureThread* capture = new DataCaptureThread(method, nsrc, core);
	
	if( format == std::string("chips") ) {
	    BF_TRY_RETURN_ELSE(*obj = new BFdatacapture_chips_impl(capture, ring, nsrc, src0,
		                                                      buffer_ntime, slot_ntime,
		                                                      sequence_callback),
		                   *obj = 0);
    } else if( format == std::string("cor") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFdatacapture_cor_impl(capture, ring, nsrc, src0,
                                                            buffer_ntime, slot_ntime,
                                                            sequence_callback),
                           *obj = 0);
    } else if( format == std::string("tbn") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFdatacapture_tbn_impl(capture, ring, nsrc, src0,
		                                                    buffer_ntime, slot_ntime,
		                                                    sequence_callback),
		                   *obj = 0);
	} else if( format == std::string("drx") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFdatacapture_drx_impl(capture, ring, nsrc, src0,
		                                                    buffer_ntime, slot_ntime,
		                                                    sequence_callback),
		                   *obj = 0);
	} else {
		return BF_STATUS_UNSUPPORTED;
	}
}

