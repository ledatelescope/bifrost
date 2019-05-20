/*
 * Copyright (c) 2016, The Bifrost Authors. All rights reserved.
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
#include <bifrost/udp_capture.h>
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
	int                    _fd;
	AlignedBuffer<uint8_t> _buf;
#if BF_VMA_ENABLED
	VMAReceiver            _vma;
#endif
public:
	UDPPacketReceiver(int fd, size_t pkt_size_max=JUMBO_FRAME_SIZE)
		: DataCaptureMethod(fd, pkt_size_max), _fd(fd), _buf(pkt_size_max)
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
};

class UDPCaptureThread : public DataCaptureThread {
	UDPPacketReceiver _udp;
	PacketStats       _stats;
	std::vector<PacketStats> _src_stats;
	bool              _have_pkt;
	PacketDesc        _pkt;
public:
	enum {
		CAPTURE_SUCCESS     = 1 << 0,
		CAPTURE_TIMEOUT     = 1 << 1,
		CAPTURE_INTERRUPTED = 1 << 2,
		CAPTURE_ERROR       = 1 << 3
	};
	UDPCaptureThread(int fd, int nsrc, int core=0, size_t pkt_size_max=9000)
		: DataCaptureThread(fd, nsrc, core, pkt_size_max), _udp(fd, pkt_size_max), _src_stats(nsrc),
		  _have_pkt(false) {
		this->reset_stats();
	}
	// Captures, decodes and unpacks packets into the provided buffers
	// Note: Capture continues until the first packet that belongs
	//         beyond the end of the provided buffers. This packet is
	//         saved, accessible via get_last_packet(), and will be
	//         processed on the next call to run() if possible.
	template<class PacketDecoder, class PacketProcessor>
	int run(uint64_t         seq_beg,
	        uint64_t         nseq_per_obuf,
	        int              nbuf,
	        uint8_t*         obufs[],
	        size_t*          ngood_bytes[],
	        size_t*          src_ngood_bytes[],
	        PacketDecoder*   decode,
	        PacketProcessor* process) {
		uint64_t seq_end = seq_beg + nbuf*nseq_per_obuf;
		size_t local_ngood_bytes[2] = {0, 0};
		int ret;
		while( true ) {
			if( !_have_pkt ) {
				uint8_t* pkt_ptr;
				int pkt_size = _udp.recv_packet(&pkt_ptr);
				if( pkt_size <= 0 ) {
					if( errno == EAGAIN || errno == EWOULDBLOCK ) {
						ret = CAPTURE_TIMEOUT; // Timed out
					} else if( errno == EINTR ) {
						ret = CAPTURE_INTERRUPTED; // Interrupted by signal
					} else {
						ret = CAPTURE_ERROR; // Socket error
					}
					break;
				}
				if( !(*decode)(pkt_ptr, pkt_size, &_pkt) ) {
					++_stats.ninvalid;
					_stats.ninvalid_bytes += pkt_size;
					continue;
				}
				_have_pkt = true;
			}
			if( greater_equal(_pkt.seq, seq_end) ) {
				// Reached the end of this processing gulp, so leave this
				//   packet unprocessed and return.
				ret = CAPTURE_SUCCESS;
				break;
			}
			_have_pkt = false;
			if( less_than(_pkt.seq, seq_beg) ) {
				++_stats.nlate;
				_stats.nlate_bytes += _pkt.payload_size;
				++_src_stats[_pkt.src].nlate;
				_src_stats[_pkt.src].nlate_bytes += _pkt.payload_size;
				continue;
			}
			++_stats.nvalid;
			_stats.nvalid_bytes += _pkt.payload_size;
			++_src_stats[_pkt.src].nvalid;
			_src_stats[_pkt.src].nvalid_bytes += _pkt.payload_size;
			// HACK TODO: src_ngood_bytes should be accumulated locally and
			//              then atomically updated, like ngood_bytes. The
			//              current way is not thread-safe.
			(*process)(&_pkt, seq_beg, nseq_per_obuf, nbuf, obufs,
			           local_ngood_bytes, /*local_*/src_ngood_bytes);
		}
		if( nbuf > 0 ) { atomic_add_and_fetch(ngood_bytes[0], local_ngood_bytes[0]); }
		if( nbuf > 1 ) { atomic_add_and_fetch(ngood_bytes[1], local_ngood_bytes[1]); }
		return ret;
	}
};

typedef int (*BFudpcapture_chips_sequence_callback)(BFoffset, int, int, int,
                                                    BFoffset*, void const**, size_t*);

class BFudpcapture_chips_impl : public BFdatacapture_impl {
	UDPCaptureThread   _capture;
	CHIPSDecoder       _decoder;
	CHIPSProcessor     _processor;
	ProcLog            _type_log;
	ProcLog            _chan_log;
	
	BFudpcapture_chips_sequence_callback _sequence_callback;
	
	void on_sequence_start(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size ) {
        // TODO: Might be safer to round to nearest here, but the current firmware
		//         always starts things ~3 seq's before the 1sec boundary anyway.
		//seq = round_up(pkt->seq, _slot_ntime);
		//*_seq          = round_nearest(pkt->seq, _slot_ntime);
		_seq          = round_up(pkt->seq, _slot_ntime);
		this->on_sequence_changed(pkt, seq0, time_tag, hdr, hdr_size);
    }
    void on_sequence_active(const PacketDesc* pkt) {
        if( pkt ) {
		    //cout << "Latest nchan, chan0 = " << pkt->nchan << ", " << pkt->chan0 << endl;
		}
		else {
			//cout << "No latest packet" << endl;
		}
	}
	inline bool has_sequence_changed(const PacketDesc* pkt) {
	    return (pkt->chan0 != _chan0) \
	           || (pkt->nchan != _nchan);
	}
	void on_sequence_changed(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size) {
	    *seq0 = _seq;// + _nseq_per_buf*_bufs.size();
	    if( _sequence_callback ) {
	        int status = (*_sequence_callback)(*seq0,
			                                   _chan0,
			                                   _nchan,
			                                   _nsrc,
			                                   time_tag,
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
        
	    _chan0 = pkt->chan0;
		_nchan = pkt->nchan;
		_payload_size = pkt->payload_size;
		_chan_log.update() << "chan0        : " << _chan0 << "\n"
		                   << "nchan        : " << _nchan << "\n"
		                   << "payload_size : " << _payload_size << "\n";
    }
public:
	inline BFudpcapture_chips_impl(int    fd,
	                               BFring ring,
	                               int    nsrc,
	                               int    src0,
	                               int    max_payload_size,
	                               int    buffer_ntime,
	                               int    slot_ntime,
	                               BFudpcapture_chips_sequence_callback sequence_callback,
	                               int    core)
		: BFdatacapture_impl("udp_capture", fd, ring, nsrc, src0, max_payload_size, buffer_ntime, slot_ntime, core), 
		  _capture(fd, nsrc, core), _decoder(nsrc, src0), _processor(),
		  _type_log("udp_capture/type"),
		  _chan_log("udp_capture/chans"),
		  _sequence_callback(sequence_callback) {
		size_t contig_span  = this->bufsize(max_payload_size);
		// Note: 2 write bufs may be open for writing at one time
		size_t total_span   = contig_span * 4;
		size_t nringlet_max = 1;
		_ring.resize(contig_span, total_span, nringlet_max);
		_type_log.update("type : %s", "chips");
	}
};

typedef int (*BFudpcapture_tbn_sequence_callback)(BFoffset, int, int, int,
                                                  BFoffset*, void const**, size_t*);

class BFudpcapture_tbn_impl : public BFdatacapture_impl {
	UDPCaptureThread _capture;
	TBNDecoder       _decoder;
	TBNProcessor     _processor;
	ProcLog          _type_log;
	ProcLog          _chan_log;
	
	BFudpcapture_tbn_sequence_callback _sequence_callback;
	
	BFoffset _time_tag;
	
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
	}
	inline bool has_sequence_changed(const PacketDesc* pkt) {
	    return (pkt->tuning != _chan0);
	}
	void on_sequence_changed(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size) {
	    *seq0 = _seq;// + _nseq_per_buf*_bufs.size();
	    if( _sequence_callback ) {
	        int status = (*_sequence_callback)(*seq0,
			                                   _chan0,
			                                   _nchan,
			                                   _nsrc,
			                                   time_tag,
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
	inline BFudpcapture_tbn_impl(int    fd,
	                             BFring ring,
	                             int    nsrc,
	                             int    src0,
	                             int    max_payload_size,
	                             int    buffer_ntime,
	                             int    slot_ntime,
	                             BFudpcapture_chips_sequence_callback sequence_callback,
	                             int    core)
		: BFdatacapture_impl("udp_capture", fd, ring, nsrc, src0, max_payload_size, buffer_ntime, slot_ntime, core), 
		  _capture(fd, nsrc, core), _decoder(nsrc, src0), _processor(),
		  _type_log("udp_capture/type"),
		  _chan_log("udp_capture/chans"),
		  _sequence_callback(sequence_callback) {
		size_t contig_span  = this->bufsize(max_payload_size);
		// Note: 2 write bufs may be open for writing at one time
		size_t total_span   = contig_span * 4;
		size_t nringlet_max = 1;
		_ring.resize(contig_span, total_span, nringlet_max);
		_type_log.update("type : %s", "tbn");
	}
};

typedef int (*BFudpcapture_drx_sequence_callback)(BFoffset, int, int, int,
                                                  BFoffset*, void const**, size_t*);
                                                    
class BFudpcapture_drx_impl : public BFdatacapture_impl {
	UDPCaptureThread _capture;
	DRXDecoder       _decoder;
	DRXProcessor     _processor;
	ProcLog          _type_log;
	ProcLog          _chan_log;
	
	BFudpcapture_drx_sequence_callback _sequence_callback;
	
	BFoffset _time_tag;
	int      _chan1;
	uint8_t  _tstate;
	
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
	    if( _sequence_callback ) {
	        int status = (*_sequence_callback)(*seq0,
			                                   _chan0,
			                                   _nchan,
			                                   _nsrc,
			                                   time_tag,
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
	inline BFudpcapture_drx_impl(int    fd,
	                             BFring ring,
	                             int    nsrc,
	                             int    src0,
	                             int    max_payload_size,
	                             int    buffer_ntime,
	                             int    slot_ntime,
	                             BFudpcapture_chips_sequence_callback sequence_callback,
	                             int    core)
		: BFdatacapture_impl("udp_capture", fd, ring, nsrc, src0, max_payload_size, buffer_ntime, slot_ntime, core), 
		  _capture(fd, nsrc, core), _decoder(nsrc, src0), _processor(),
		  _type_log("udp_capture/type"),
		  _chan_log("udp_capture/chans"),
		  _sequence_callback(sequence_callback), 
		  _chan1(), _tstate(0) {
		size_t contig_span  = this->bufsize(max_payload_size);
		// Note: 2 write bufs may be open for writing at one time
		size_t total_span   = contig_span * 4;
		size_t nringlet_max = 1;
		_ring.resize(contig_span, total_span, nringlet_max);
		_type_log.update("type : %s", "tbn");
	}
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
                            BFudpcapture_sequence_callback sequence_callback,
                            int            core) {
	BF_ASSERT(obj, BF_STATUS_INVALID_POINTER);
	if( format == std::string("chips") ) {
		BF_TRY_RETURN_ELSE(*obj = new BFudpcapture_chips_impl(fd, ring, nsrc, src0, max_payload_size,
		                                                      buffer_ntime, slot_ntime,
		                                                      sequence_callback, core),
		                   *obj = 0);
    } else if( format == std::string("tbn") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFudpcapture_tbn_impl(fd, ring, nsrc, src0, max_payload_size,
		                                                    buffer_ntime, slot_ntime,
		                                                    sequence_callback, core),
		                   *obj = 0);
	} else if( format == std::string("drx") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFudpcapture_drx_impl(fd, ring, nsrc, src0, max_payload_size,
		                                                    buffer_ntime, slot_ntime,
		                                                    sequence_callback, core),
		                   *obj = 0);
	} else {
		return BF_STATUS_UNSUPPORTED;
	}
}

