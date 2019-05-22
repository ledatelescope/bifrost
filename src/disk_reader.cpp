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

class DiskPacketReader : virtual public DataCaptureMethod {
public:
	DiskPacketReader(int fd, size_t pkt_size_max=9000)
	 : DataCaptureMethod(fd, pkt_size_max) {}
	int recv_packet(uint8_t** pkt_ptr, int flags=0) {
		*pkt_ptr = &_buf[0];
		return ::read(_fd, &_buf[0], _buf.size());
	}
};

class DiskReaderThread : virtual public DataCaptureThread {
	DiskPacketReader  _method;
	
	DataCaptureMethod* get_method() {
	    return &_method;
	}
public:
	DiskReaderThread(int fd, int nsrc, int core=0, size_t pkt_size_max=9000)
	 : DataCaptureThread(fd, nsrc, core, pkt_size_max), _method(fd, pkt_size_max) {
	    this->reset_stats();
	}
};

class BFdiskreader_tbn_impl : virtual public BFdatacapture_impl {
	DiskReaderThread _capture;
	TBNDecoder       _decoder;
	TBNProcessor     _processor;
	ProcLog          _type_log;
	ProcLog          _chan_log;
	
	BFdatacapture_tbn_sequence_callback _sequence_callback;
	
	BFoffset _time_tag;
	
	inline DataCaptureThread* get_capture() {
	    return &_capture;
	}
	inline PacketDecoder* get_decoder() {
	    return &_decoder;
	}
	inline PacketProcessor* get_processor() {
	    return &_processor;
    }
	void on_sequence_start(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size ) {
		_seq          = round_nearest(pkt->seq, _nseq_per_buf);
		this->on_sequence_changed(pkt, seq0, time_tag, hdr, hdr_size);
    }
    void on_sequence_active(const PacketDesc* pkt) {
        if( pkt ) {
		    //cout << "Latest time_tag, tuning = " << pkt->time_tag << ", " << pkt->tuning << endl;
		}
		else {
			//cout << "No latest packet" << endl;
		}
	}
	inline bool has_sequence_changed(const PacketDesc* pkt) {
	    return (pkt->tuning != _chan0);
	}
	void on_sequence_changed(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size) {
	    //cout << "Sequence changed" << endl;
	    *seq0 = _seq;// + _nseq_per_buf*_bufs.size();
	    *time_tag = pkt->time_tag;
	    if( _sequence_callback ) {
	        int status = (*_sequence_callback)(*seq0,
	                                        *time_tag,
			                                _chan0,
			                                _nsrc,
			                                hdr,
			                                hdr_size);
			if( status != 0 ) {
			    // TODO: What to do here? Needed?
				throw std::runtime_error("BAD HEADER CALLBACK STATUS");
			}
		} else {
			// Simple default for easy testing
			*time_tag = *seq0;
			*hdr      = NULL;
			*hdr_size = 0;
		}
        
	    _time_tag     = pkt->time_tag;
		_chan0        = pkt->tuning;
		_payload_size = pkt->payload_size;
		_chan_log.update() << "chan0        : " << _chan0 << "\n"
				           << "payload_size : " << _payload_size << "\n";
    }
public:
	inline BFdiskreader_tbn_impl(int    fd,
	                             BFring ring,
	                             int    nsrc,
	                             int    src0,
	                             int    buffer_ntime,
	                             int    slot_ntime,
	                             BFdatacapture_callback sequence_callback,
	                             int    core)
		: BFdatacapture_impl("disk_reader", fd, ring, nsrc, src0, TBN_FRAME_SIZE, buffer_ntime, slot_ntime, core),
		  _capture(fd, nsrc, core, TBN_FRAME_SIZE), _decoder(nsrc, src0), _processor(),
		  _type_log("disk_reader/type"),
		  _chan_log("disk_reader/chans"),
		  _sequence_callback(sequence_callback->get_tbn()) {
		_type_log.update("type : %s\n", "tbn");
	}
};
                                                    
class BFdiskreader_drx_impl : virtual public BFdatacapture_impl {
	DiskReaderThread _capture;
	DRXDecoder       _decoder;
	DRXProcessor     _processor;
	ProcLog          _type_log;
	ProcLog          _chan_log;
	
	BFdatacapture_drx_sequence_callback _sequence_callback;
	
	BFoffset _time_tag;
	int      _chan1;
	uint8_t  _tstate;
	
	inline DataCaptureThread* get_capture() {
	    return &_capture;
	}
	inline PacketDecoder* get_decoder() {
	    return &_decoder;
	}
	inline PacketProcessor* get_processor() {
	    return &_processor;
    }
	void on_sequence_start(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size ) {
		_seq          = round_nearest(pkt->seq, _nseq_per_buf);
		this->on_sequence_changed(pkt, seq0, time_tag, hdr, hdr_size);
    }
    void on_sequence_active(const PacketDesc* pkt) {
        if( pkt ) {
		    //cout << "Latest nchan, chan0 = " << pkt->nchan << ", " << pkt->chan0 << endl;
		}
		else {
			//cout << "No latest packet" << endl;
		}
		
		if( pkt->src / 2  == 0 ) {
			if( pkt->tuning != _chan0 ) {
				_tstate |= 1;
				_chan0 = pkt->tuning;
			}
		} else {
			if( pkt->tuning != _chan1 ) {
				_tstate |= 2;
				_chan1 = pkt->tuning;
			}
		}
	}
	inline bool has_sequence_changed(const PacketDesc* pkt) {
	    return (   (_tstate == 3 && _nsrc == 4) 
	            || (_tstate != 0 && _nsrc == 2) );
	}
	void on_sequence_changed(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size) {
	    *seq0 = _seq;// + _nseq_per_buf*_bufs.size();
	    *time_tag = pkt->time_tag;
	    if( _sequence_callback ) {
	        int status = (*_sequence_callback)(*seq0,
	                                        *time_tag,
			                                _chan0,
			                                _chan1,
			                                _nsrc,
			                                hdr,
			                                hdr_size);
			if( status != 0 ) {
			    // TODO: What to do here? Needed?
				throw std::runtime_error("BAD HEADER CALLBACK STATUS");
			}
		} else {
			// Simple default for easy testing
			*time_tag = *seq0;
			*hdr      = NULL;
			*hdr_size = 0;
		}
        
	    _time_tag     = pkt->time_tag;
		_payload_size = pkt->payload_size;
		_chan_log.update() << "chan0        : " << _chan0 << "\n"
					       << "chan1        : " << _chan1 << "\n"
					       << "payload_size : " << _payload_size << "\n";
    }
public:
	inline BFdiskreader_drx_impl(int    fd,
	                             BFring ring,
	                             int    nsrc,
	                             int    src0,
	                             int    buffer_ntime,
	                             int    slot_ntime,
	                             BFdatacapture_callback sequence_callback,
	                             int    core)
		: BFdatacapture_impl("disk_reader", fd, ring, nsrc, src0, DRX_FRAME_SIZE, buffer_ntime, slot_ntime, core), 
		  _capture(fd, nsrc, core, DRX_FRAME_SIZE), _decoder(nsrc, src0), _processor(),
		  _type_log("disk_reader/type"),
		  _chan_log("disk_reader/chans"),
		  _sequence_callback(sequence_callback->get_drx()), 
		  _chan1(), _tstate(0) {
		_type_log.update("type : %s\n", "drx");
	}
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
	if( format == std::string("tbn") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFdiskreader_tbn_impl(fd, ring, nsrc, src0,
		                                                    buffer_ntime, slot_ntime,
		                                                    sequence_callback, core),
		                   *obj = 0);
	} else if( format == std::string("drx") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFdiskreader_drx_impl(fd, ring, nsrc, src0,
		                                                    buffer_ntime, slot_ntime,
		                                                    sequence_callback, core),
		                   *obj = 0);
	} else {
		return BF_STATUS_UNSUPPORTED;
	}
}

