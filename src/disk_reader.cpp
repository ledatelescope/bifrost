/*
 * Copyright (c) 2017, The Bifrost Authors. All rights reserved.
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
#include <bifrost/data_capture.h>
#include <bifrost/disk_reader.h>
#include <bifrost/affinity.h>
#include <bifrost/Ring.hpp>
using bifrost::ring::RingWrapper;
using bifrost::ring::RingWriter;
using bifrost::ring::WriteSpan;
using bifrost::ring::WriteSequence;
#include "proclog.hpp"
#include "formats/formats.hpp"
#include "data_capture.hpp"

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

class DiskPacketReader : public DataCaptureMethod {
public:
	DiskPacketReader(int fd, size_t pkt_size_max=9000)
	 : DataCaptureMethod(fd, pkt_size_max) {}
	int recv_packet(uint8_t** pkt_ptr, int flags=0) {
		*pkt_ptr = &_buf[0];
		return ::read(_fd, &_buf[0], _buf.size());
	}
	inline const char* get_name() { return "disk_reader"; }
};

BFstatus bfDiskReaderCreate(BFdatacapture* obj,
                            const char*    format,
                            int            fd,
                            BFring         ring,
                            BFsize         nsrc,
                            BFsize         src0,
                            BFsize         buffer_ntime,
                            BFsize         slot_ntime,
                            BFdatacapture_callback sequence_callback,
                            int            core) {
	BF_ASSERT(obj, BF_STATUS_INVALID_POINTER);
	
	size_t max_payload_size = JUMBO_FRAME_SIZE;
	if( std::string(format).substr(0, 6) == std::string("chips_") ) {
	    int nchan = std::atoi((std::string(format).substr(6, std::string(format).length())).c_str());
	    max_payload_size = sizeof(chips_hdr_type) + 32*nchan;
	} else if( format == std::string("tbn") ) {
	    max_payload_size = TBN_FRAME_SIZE;
	} else if( format == std::string("drx") ) {
	    max_payload_size = DRX_FRAME_SIZE;
	}
	
	DiskPacketReader* method = new DiskPacketReader(fd, max_payload_size);
	DataCaptureThread* capture = new DataCaptureThread(method, nsrc, core);
	
	if( std::string(format).substr(0, 6) == std::string("chips_") ) {
	    BF_TRY_RETURN_ELSE(*obj = new BFdatacapture_chips_impl(capture, ring, nsrc, src0,
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

