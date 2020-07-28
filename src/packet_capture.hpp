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

#include "assert.hpp"
#include <bifrost/io.h>
#include <bifrost/packet_capture.h>
#include <bifrost/Ring.hpp>
using bifrost::ring::RingWrapper;
using bifrost::ring::RingWriter;
using bifrost::ring::WriteSpan;
using bifrost::ring::WriteSequence;
#include "proclog.hpp"
#include "formats/formats.hpp"
#include "hw_locality.hpp"

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

#ifndef BF_HWLOC_ENABLED
#define BF_HWLOC_ENABLED 0
//#define BF_HWLOC_ENABLED 1
#endif

template<typename T>
inline T atomic_add_and_fetch(T* dst, T val) {
	return __sync_add_and_fetch(dst, val); // GCC builtin
}
template<typename T>
inline T atomic_fetch_and_add(T* dst, T val) {
	return __sync_fetch_and_add(dst, val); // GCC builtin
}

// Wrap-safe comparisons
inline bool greater_equal(uint64_t a, uint64_t b) { return int64_t(a-b) >= 0; }
inline bool less_than(    uint64_t a, uint64_t b) { return int64_t(a-b) <  0; }

template<typename T>
class AlignedBuffer {
	enum { DEFAULT_ALIGNMENT = 4096 };
	T*     _buf;
	size_t _size;
	size_t _alignment;
	void alloc() {
		int err = ::posix_memalign((void**)&_buf, _alignment, _size*sizeof(T));
		if( err ) {
			throw std::runtime_error("Allocation failed");
		}
	}
	void copy(T const* srcbuf, size_t n) {
		::memcpy(_buf, srcbuf, n*sizeof(T));
	}
	void free() {
		if( _buf ) {
			::free(_buf);
			_buf  = 0;
			_size = 0;
		}
	}
public:
	//explicit AlignedBuffer(size_t alignment=DEFAULT_ALIGNMENT)
	//	: _buf(0), _size(0), _alignment(alignment) {}
	AlignedBuffer(size_t size=0, size_t alignment=DEFAULT_ALIGNMENT)
		: _buf(0), _size(size), _alignment(alignment) {
		this->alloc();
	}
	AlignedBuffer(AlignedBuffer const& other)
		: _buf(0), _size(other.size), _alignment(other.alignment) {
		this->alloc();
		this->copy(other._buf, other._size);
	}
    AlignedBuffer& operator=(AlignedBuffer const& other) {
	    if( &other != this ) {
		    this->free();
		    _size      = other.size;
		    _alignment = other.alignment;
		    this->alloc();
		    this->copy(other._buf, other._size);
	    }
	    return *this;
    }
	~AlignedBuffer() {
		this->free();
	}
	inline void swap(AlignedBuffer & other) {
		std::swap(_buf,       other._buf);
		std::swap(_size,      other._size);
		std::swap(_alignment, other._alignment);
	}
	inline void resize(size_t n) {
		if( n <= _size ) {
			_size = n;
		} else {
			AlignedBuffer tmp(n, _alignment);
			tmp.copy(&_buf[0], _size);
			tmp.swap(*this);
		}
	}
	inline size_t size() const                 { return _size;      }
	inline size_t alignment() const            { return _alignment; }
	inline T      & operator[](size_t i)       { return _buf[i];    }
	inline T const& operator[](size_t i) const { return _buf[i];    }
};

class PacketCaptureMethod {
protected:
    int                    _fd;
    size_t                 _pkt_size_max;
	AlignedBuffer<uint8_t> _buf;
    BFiomethod             _io_method;
public:
	PacketCaptureMethod(int fd, size_t pkt_size_max=9000, BFiomethod io_method=BF_IO_GENERIC)
	 : _fd(fd), _pkt_size_max(pkt_size_max), _buf(pkt_size_max), _io_method(io_method) {}
	virtual int recv_packet(uint8_t** pkt_ptr, int flags=0) {
	    return 0;
	}
	virtual const char* get_name() { return "generic_capture"; }
	inline const size_t get_max_size() { return _pkt_size_max; }
	inline const BFiomethod get_io_method() { return _io_method; }
	virtual inline BFoffset seek(BFoffset offset, BFiowhence whence=BF_WHENCE_CUR) { return 0; }
	inline BFoffset tell() { return this->seek(0, BF_WHENCE_CUR); }
};

class DiskPacketReader : public PacketCaptureMethod {
public:
    DiskPacketReader(int fd, size_t pkt_size_max=9000)
     : PacketCaptureMethod(fd, pkt_size_max, BF_IO_DISK) {}
    int recv_packet(uint8_t** pkt_ptr, int flags=0) {
        *pkt_ptr = &_buf[0];
        return ::read(_fd, &_buf[0], _buf.size());
    }
    inline const char* get_name() { return "disk_reader"; }
    inline BFoffset seek(BFoffset offset, BFiowhence whence=BF_WHENCE_CUR) {
        return ::lseek64(_fd, offset, whence);
    }
};

// TODO: The VMA API is returning unaligned buffers, which prevents use of SSE
#ifndef BF_VMA_ENABLED
#define BF_VMA_ENABLED 0
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

class UDPPacketReceiver : public PacketCaptureMethod {
#if BF_VMA_ENABLED
    VMAReceiver            _vma;
#endif
public:
    UDPPacketReceiver(int fd, size_t pkt_size_max=JUMBO_FRAME_SIZE)
        : PacketCaptureMethod(fd, pkt_size_max, BF_IO_UDP)
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

class UDPPacketSniffer : public PacketCaptureMethod {
#if BF_VMA_ENABLED
    VMAReceiver            _vma;
#endif
public:
    UDPPacketSniffer(int fd, size_t pkt_size_max=JUMBO_FRAME_SIZE)
        : PacketCaptureMethod(fd, pkt_size_max, BF_IO_SNIFFER)
#if BF_VMA_ENABLED
        , _vma(fd)
#endif
    {}
    inline int recv_packet(uint8_t** pkt_ptr, int flags=0) {
#if BF_VMA_ENABLED
        if( _vma ) {
            *pkt_ptr = 0;
            int rc = _vma.recv_packet(&_buf[0], _buf.size(), pkt_ptr, flags) - 28;
            *pkt_ptr = *(pkt_ptr + 28);
            return rc;
        } else {
#endif
            *pkt_ptr = &_buf[28];   // Offset for the IP+UDP headers
            return ::recvfrom(_fd, &_buf[0], _buf.size(), flags, 0, 0) - 28;
#if BF_VMA_ENABLED
        }
#endif
    }
    inline const char* get_name() { return "udp_sniffer"; }
};

#ifndef BF_VERBS_ENABLED
#define BF_VERBS_ENABLED 0
#endif

#if BF_VERBS_ENABLED
#include "ib_verbs.hpp"

class UDPVerbsReceiver : public PacketCaptureMethod {
    Verbs                  _ibv;
public:
    UDPVerbsReceiver(int fd, size_t pkt_size_max=JUMBO_FRAME_SIZE, int core=-1)
        : PacketCaptureMethod(fd, pkt_size_max, BF_IO_VERBS), _ibv(fd, pkt_size_max, core) {}
    inline int recv_packet(uint8_t** pkt_ptr, int flags=0) {
        *pkt_ptr = 0;
        return _ibv.recv_packet(pkt_ptr, flags);
    }
    inline const char* get_name() { return "udp_verbs_capture"; }
};
#endif // BF_VERBS_ENABLED

struct PacketStats {
	size_t ninvalid;
	size_t ninvalid_bytes;
	size_t nlate;
	size_t nlate_bytes;
	size_t nvalid;
	size_t nvalid_bytes;
};

class PacketCaptureThread : public BoundThread {
private:
    PacketCaptureMethod*     _method;
    PacketStats              _stats;
	std::vector<PacketStats> _src_stats;
	bool                     _have_pkt;
	PacketDesc               _pkt;
	int                      _core;
    
public:
	enum {
		CAPTURE_SUCCESS     = 1 << 0,
		CAPTURE_TIMEOUT     = 1 << 1,
		CAPTURE_INTERRUPTED = 1 << 2,
		CAPTURE_NO_DATA     = 1 << 3,
		CAPTURE_ERROR       = 1 << 4
	};
	PacketCaptureThread(PacketCaptureMethod* method, int nsrc, int core=0)
     : BoundThread(core), _method(method), _src_stats(nsrc),
	   _have_pkt(false), _core(core) {
		this->reset_stats();
	}
	template<class PDC, class PPC>
	int run(uint64_t seq_beg,
	        uint64_t nseq_per_obuf,
	        int      nbuf,
	        uint8_t* obufs[],
	        size_t*  ngood_bytes[],
	        size_t*  src_ngood_bytes[],
	        PDC*     decode,
	        PPC*     process);
	inline const char* get_name() { return _method->get_name(); }
	inline const size_t get_max_size() { return _method->get_max_size(); }
	inline const BFiomethod get_io_method() { return _method->get_io_method(); }
	inline const int get_core() { return _core; }
	inline BFoffset seek(BFoffset offset, BFiowhence whence=BF_WHENCE_CUR) { return _method->seek(offset, whence); }
	inline BFoffset tell() { return _method->tell(); }
    inline const PacketDesc* get_last_packet() const {
		return _have_pkt ? &_pkt : NULL;
	}
	inline void reset_last_packet() {
		_have_pkt = false;
	}
	inline const PacketStats* get_stats() const { return &_stats; }
	inline const PacketStats* get_stats(int src) const { return &_src_stats[src]; }
	inline void reset_stats() {
		::memset(&_stats, 0, sizeof(_stats));
		::memset(&_src_stats[0], 0, _src_stats.size()*sizeof(PacketStats));
	}
};

inline uint64_t round_up(uint64_t val, uint64_t mult) {
	return (val == 0 ?
	        0 :
	        ((val-1)/mult+1)*mult);
}
inline uint64_t round_nearest(uint64_t val, uint64_t mult) {
	return (2*val/mult+1)/2*mult;
}

class BFpacketcapture_callback_impl {
    BFpacketcapture_chips_sequence_callback _chips_callback;
    BFpacketcapture_ibeam_sequence_callback _ibeam_callback;
    BFpacketcapture_cor_sequence_callback   _cor_callback;
    BFpacketcapture_vdif_sequence_callback  _vdif_callback;
    BFpacketcapture_tbn_sequence_callback   _tbn_callback;
    BFpacketcapture_drx_sequence_callback   _drx_callback;
public:
    BFpacketcapture_callback_impl()
     : _chips_callback(NULL), _ibeam_callback(NULL), _cor_callback(NULL), 
        _vdif_callback(NULL), _tbn_callback(NULL), _drx_callback(NULL) {}
    inline void set_chips(BFpacketcapture_chips_sequence_callback callback) {
        _chips_callback = callback;
    }
    inline BFpacketcapture_chips_sequence_callback get_chips() {
        return _chips_callback;
    }
    inline void set_ibeam(BFpacketcapture_ibeam_sequence_callback callback) {
        _ibeam_callback = callback;
    }
    inline BFpacketcapture_ibeam_sequence_callback get_ibeam() {
        return _ibeam_callback;
    }
    inline void set_cor(BFpacketcapture_cor_sequence_callback callback) {
        _cor_callback = callback;
    }
    inline BFpacketcapture_cor_sequence_callback get_cor() {
        return _cor_callback;
    }
    inline void set_vdif(BFpacketcapture_vdif_sequence_callback callback) {
        _vdif_callback = callback;
    }
    inline BFpacketcapture_vdif_sequence_callback get_vdif() {
        return _vdif_callback;
    }
    inline void set_tbn(BFpacketcapture_tbn_sequence_callback callback) {
        _tbn_callback = callback;
    }
    inline BFpacketcapture_tbn_sequence_callback get_tbn() {
        return _tbn_callback;
    }
    inline void set_drx(BFpacketcapture_drx_sequence_callback callback) {
        _drx_callback = callback;
    }
    inline BFpacketcapture_drx_sequence_callback get_drx() {
        return _drx_callback;
    }
};

class BFpacketcapture_impl {
protected:
    std::string          _name;
	PacketCaptureThread* _capture;
	PacketDecoder*       _decoder;
	PacketProcessor*     _processor;
	ProcLog              _bind_log;
	ProcLog              _out_log;
	ProcLog              _size_log;
	ProcLog              _stat_log;
	ProcLog              _perf_log;
	pid_t                _pid;
	
	int                  _nsrc;
	int                  _nseq_per_buf;
	int                  _slot_ntime;
	BFoffset             _seq;
	int                  _chan0;
	int                  _nchan;
	int                  _payload_size;
	bool                 _active;

private:
    std::chrono::high_resolution_clock::time_point _t0;
	std::chrono::high_resolution_clock::time_point _t1;
	std::chrono::high_resolution_clock::time_point _t2;
	std::chrono::duration<double> _process_time;
	std::chrono::duration<double> _reserve_time;
	
	RingWrapper _ring;
	RingWriter  _oring;
	std::queue<std::shared_ptr<WriteSpan> > _bufs;
	std::queue<size_t>                      _buf_ngood_bytes;
	std::queue<std::vector<size_t> >        _buf_src_ngood_bytes;
	std::shared_ptr<WriteSequence>          _sequence;
	size_t _ngood_bytes;
	size_t _nmissing_bytes;
	
	inline size_t bufsize(int payload_size=-1) {
		if( payload_size == -1 ) {
			payload_size = _payload_size;
		}
		return _nseq_per_buf * _nsrc * payload_size * BF_UNPACK_FACTOR;
	}
	inline void reserve_buf() {
		_buf_ngood_bytes.push(0);
		_buf_src_ngood_bytes.push(std::vector<size_t>(_nsrc, 0));
		size_t size = this->bufsize();
		// TODO: Can make this simpler?
		_bufs.push(std::shared_ptr<WriteSpan>(new bifrost::ring::WriteSpan(_oring, size)));
	}
	inline void commit_buf() {
		size_t expected_bytes = _bufs.front()->size();
		
		for( int src=0; src<_nsrc; ++src ) {
			// TODO: This assumes all sources contribute equally; should really
			//         allow non-uniform partitioning.
			size_t src_expected_bytes = expected_bytes / _nsrc;
			size_t src_ngood_bytes    = _buf_src_ngood_bytes.front()[src];
			size_t src_nmissing_bytes = src_expected_bytes - src_ngood_bytes;
			// Detect >50% missing data from this source
			if( src_nmissing_bytes > src_ngood_bytes ) {
				// Zero-out this source's contribution to the buffer
				uint8_t* data = (uint8_t*)_bufs.front()->data();
				_processor->blank_out_source(data, src, _nsrc,
				                             _nchan, _nseq_per_buf);
			}
		}
		_buf_src_ngood_bytes.pop();
		
		_ngood_bytes    += _buf_ngood_bytes.front();
		//_nmissing_bytes += _bufs.front()->size() - _buf_ngood_bytes.front();
		//// HACK TESTING 15/16 correction for missing roach11
		//_nmissing_bytes += _bufs.front()->size()*15/16 - _buf_ngood_bytes.front();
		_nmissing_bytes += expected_bytes - _buf_ngood_bytes.front();
		_buf_ngood_bytes.pop();
		
		_bufs.front()->commit();
		_bufs.pop();
		_seq += _nseq_per_buf;
	}
	inline void begin_sequence(BFoffset seq0, BFoffset time_tag, const void* hdr, size_t hdr_size) {
		const char* name     = "";
		int         nringlet = 1;
		_sequence.reset(new WriteSequence(_oring, name, time_tag,
		                                  hdr_size, hdr, nringlet));
	}
	inline void end_sequence() {
		_sequence.reset(); // Note: This is releasing the shared_ptr
	}
	virtual void on_sequence_start(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size) {}
	virtual void on_sequence_active(const PacketDesc* pkt) {}
	virtual inline bool has_sequence_changed(const PacketDesc* pkt) { return false; }
	virtual void on_sequence_changed(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size) {}
public:
	inline BFpacketcapture_impl(PacketCaptureThread* capture,
	                          PacketDecoder*     decoder,
	                          PacketProcessor*   processor,
				              BFring      ring,
            	              int         nsrc,
	                          int         buffer_ntime,
	                          int         slot_ntime)
		: _name(capture->get_name()), _capture(capture), _decoder(decoder), _processor(processor),
		  _bind_log(_name+"/bind"),
		  _out_log(_name+"/out"),
		  _size_log(_name+"/sizes"),
		  _stat_log(_name+"/stats"),
		  _perf_log(_name+"/perf"), 
		  _nsrc(nsrc), _nseq_per_buf(buffer_ntime), _slot_ntime(slot_ntime),
		  _seq(), _chan0(), _nchan(), _active(false),
		  _ring(ring), _oring(_ring),
		  // TODO: Add reset method for stats
		  _ngood_bytes(0), _nmissing_bytes(0) {
        size_t contig_span  = this->bufsize(_capture->get_max_size());
		// Note: 2 write bufs may be open for writing at one time
		size_t total_span   = contig_span * 4;
		size_t nringlet_max = 1;
		_ring.resize(contig_span, total_span, nringlet_max);
		_bind_log.update("ncore : %i\n"
		                 "core0 : %i\n", 
		                 1, _capture->get_core());
		_out_log.update("nring : %i\n"
		                "ring0 : %s\n", 
		                1, _ring.name());
		_size_log.update("nsrc         : %i\n"
		                 "nseq_per_buf : %i\n"
		                 "slot_ntime   : %i\n",
		                 _nsrc, _nseq_per_buf, _slot_ntime);
	}
    virtual ~BFpacketcapture_impl() {}
    inline void flush() {
		while( _bufs.size() ) {
			this->commit_buf();
		}
		if( _sequence ) {
			this->end_sequence();
		}
	}
	inline BFoffset seek(BFoffset offset, BFiowhence whence=BF_WHENCE_CUR) {
        BF_ASSERT(_capture->get_io_method() == BF_IO_DISK, BF_STATUS_UNSUPPORTED);
        BFoffset moved = _capture->seek(offset, whence);
        this->flush();
        return moved;
    }
    inline BFoffset tell() {
        BF_ASSERT(_capture->get_io_method() == BF_IO_DISK, BF_STATUS_UNSUPPORTED);
        return _capture->tell();
    }
	inline void end_writing() {
		this->flush();
		_oring.close();
	}
	BFpacketcapture_status recv();
};

class BFpacketcapture_chips_impl : public BFpacketcapture_impl {
	ProcLog            _type_log;
	ProcLog            _chan_log;
	
	BFpacketcapture_chips_sequence_callback _sequence_callback;
	
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
        _chan0 = pkt->chan0;
        _nchan = pkt->nchan;
        _payload_size = pkt->payload_size;
        
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
        
		_chan_log.update() << "chan0        : " << _chan0 << "\n"
		                   << "nchan        : " << _nchan << "\n"
		                   << "payload_size : " << _payload_size << "\n";
    }
public:
	inline BFpacketcapture_chips_impl(PacketCaptureThread* capture,
	                                  BFring               ring,
	                                  int                  nsrc,
	                                  int                  src0,
	                                  int                  buffer_ntime,
	                                  int                  slot_ntime,
	                                  BFpacketcapture_callback sequence_callback)
		: BFpacketcapture_impl(capture, nullptr, nullptr, ring, nsrc, buffer_ntime, slot_ntime), 
		  _type_log((std::string(capture->get_name())+"/type").c_str()),
		  _chan_log((std::string(capture->get_name())+"/chans").c_str()),
		  _sequence_callback(sequence_callback->get_chips()) {
		_decoder = new CHIPSDecoder(nsrc, src0);
		_processor = new CHIPSProcessor();
		_type_log.update("type : %s\n", "chips");
	}
};

template<uint8_t B>
class BFpacketcapture_ibeam_impl : public BFpacketcapture_impl {
    uint8_t            _nbeam = B;
    ProcLog            _type_log;
    ProcLog            _chan_log;
    
    BFpacketcapture_ibeam_sequence_callback _sequence_callback;
    
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
        _chan0 = pkt->chan0;
        _nchan = pkt->nchan;
        _payload_size = pkt->payload_size;
        
        if( _sequence_callback ) {
            int status = (*_sequence_callback)(*seq0,
                                               _chan0,
                                               _nchan*_nsrc,
                                               _nbeam,
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
        
        _chan_log.update() << "chan0        : " << _chan0 << "\n"
                           << "nchan        : " << _nchan << "\n"
                           << "payload_size : " << _payload_size << "\n";
    }
public:
    inline BFpacketcapture_ibeam_impl(PacketCaptureThread* capture,
                                      BFring               ring,
                                      int                  nsrc,
                                      int                  src0,
                                      int                  buffer_ntime,
                                      int                  slot_ntime,
                                      BFpacketcapture_callback sequence_callback)
        : BFpacketcapture_impl(capture, nullptr, nullptr, ring, nsrc, buffer_ntime, slot_ntime), 
          _type_log((std::string(capture->get_name())+"/type").c_str()),
          _chan_log((std::string(capture->get_name())+"/chans").c_str()),
          _sequence_callback(sequence_callback->get_ibeam()) {
        _decoder = new IBeamDecoder<B>(nsrc, src0);
        _processor = new IBeamProcessor<B>();
        _type_log.update("type : %s%i\n", "ibeam", _nbeam);
    }
};

class BFpacketcapture_cor_impl : public BFpacketcapture_impl {
    ProcLog          _type_log;
    ProcLog          _chan_log;
    
    BFpacketcapture_cor_sequence_callback _sequence_callback;
    
    BFoffset _time_tag;
    int      _navg;
    
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
        return ((pkt->chan0 != _chan0) \
                || (pkt->nchan != _nchan) \
                || (pkt->decimation != _navg));
    }
    void on_sequence_changed(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size) {
        *seq0 = _seq;// + _nseq_per_buf*_bufs.size();
        *time_tag = pkt->time_tag;
        _chan0 = pkt->chan0;
        _nchan = pkt->nchan;
        _navg  = pkt->decimation;
        _payload_size = pkt->payload_size;
        
        if( _sequence_callback ) {
            int status = (*_sequence_callback)(*seq0,
                                               *time_tag,
                                               _chan0,
                                               _nchan*((pkt->tuning >> 8) & 0xFF),
                                               _navg,
                                               _nsrc/((pkt->tuning >> 8) & 0xFF),
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
        
        _chan_log.update() << "chan0        : " << _chan0 << "\n"
                           << "nchan        : " << _nchan*((pkt->tuning >> 8) & 0xFF) << "\n"
                           << "payload_size : " << _payload_size << "\n";
    }
public:
    inline BFpacketcapture_cor_impl(PacketCaptureThread* capture,
                                  BFring                 ring,
                                  int                    nsrc,
                                  int                    src0,
                                  int                    buffer_ntime,
                                  int                    slot_ntime,
                                  BFpacketcapture_callback sequence_callback)
        : BFpacketcapture_impl(capture, nullptr, nullptr, ring, nsrc, buffer_ntime, slot_ntime), 
          _type_log((std::string(capture->get_name())+"/type").c_str()),
          _chan_log((std::string(capture->get_name())+"/chans").c_str()),
          _sequence_callback(sequence_callback->get_cor()) {
        _decoder = new CORDecoder(nsrc, src0);
        _processor = new CORProcessor();
        _type_log.update("type : %s\n", "cor");
    }
};

class BFpacketcapture_vdif_impl : public BFpacketcapture_impl {
    ProcLog          _type_log;
    ProcLog          _chan_log;
    
    BFpacketcapture_vdif_sequence_callback _sequence_callback;
    
    BFoffset _time_tag;
    int      _tuning;
    int      _sample_rate;
    
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
        return ((pkt->chan0 != _chan0) \
                || (pkt->nchan != _nchan) \
                || (pkt->tuning != _tuning));
    }
    void on_sequence_changed(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size) {
        *seq0 = _seq;// + _nseq_per_buf*_bufs.size();
        *time_tag = pkt->time_tag;
        _chan0 = pkt->chan0;
        _nchan = pkt->nchan;
        _tuning = pkt->tuning;
        _sample_rate = pkt->sync;
        _payload_size = pkt->payload_size;
        
        int ref_epoch  = (_tuning >> 16) & 0xFF;
        int bit_depth  = (_tuning >> 8) & 0xFF;
        int is_complex = (_tuning & 1);
        
        if( _sequence_callback ) {
            int status = (*_sequence_callback)(*seq0,
                                               *time_tag,
                                               ref_epoch,
                                               _sample_rate,
                                               _chan0,
                                               bit_depth,
                                               is_complex,
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
        
        _chan_log.update() << "nchan        : " << _chan0 << "\n"
                           << "bitdepth     : " << bit_depth << "\n"
                           << "complex      : " << is_complex << "\n"
                           << "payload_size : " << (_payload_size*8) << "\n";
    }
public:
    inline BFpacketcapture_vdif_impl(PacketCaptureThread* capture,
                                     BFring                 ring,
                                     int                    nsrc,
                                     int                    src0,
                                     int                    buffer_ntime,
                                     int                    slot_ntime,
                                     BFpacketcapture_callback sequence_callback)
        : BFpacketcapture_impl(capture, nullptr, nullptr, ring, nsrc, buffer_ntime, slot_ntime), 
          _type_log((std::string(capture->get_name())+"/type").c_str()),
          _chan_log((std::string(capture->get_name())+"/chans").c_str()),
          _sequence_callback(sequence_callback->get_vdif()) {
        _decoder = new VDIFDecoder(nsrc, src0);
        _processor = new VDIFProcessor();
        _type_log.update("type : %s\n", "vdif");
    }
};

class BFpacketcapture_tbn_impl : public BFpacketcapture_impl {
	ProcLog          _type_log;
	ProcLog          _chan_log;
	
	BFpacketcapture_tbn_sequence_callback _sequence_callback;
	
	BFoffset _time_tag;
	uint16_t _decim;
    
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
	    return (    (pkt->tuning != _chan0     )
                 || ( pkt->decimation != _decim) );
	}
	void on_sequence_changed(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size) {
	    //cout << "Sequence changed" << endl;
	    *seq0 = _seq;// + _nseq_per_buf*_bufs.size();
	    *time_tag = pkt->time_tag;
        _time_tag     = pkt->time_tag;
        _decim        = pkt->decimation;
        _chan0        = pkt->tuning;
        _payload_size = pkt->payload_size;
        
	    if( _sequence_callback ) {
	        int status = (*_sequence_callback)(*seq0,
	                                           *time_tag,
                                               _decim,
			                                   pkt->tuning,
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
        
		_chan_log.update() << "chan0        : " << _chan0 << "\n"
				           << "payload_size : " << _payload_size << "\n";
    }
public:
	inline BFpacketcapture_tbn_impl(PacketCaptureThread* capture,
	                                BFring               ring,
	                                int                  nsrc,
	                                int                  src0,
	                                int                  buffer_ntime,
	                                int                  slot_ntime,
	                                BFpacketcapture_callback sequence_callback)
		: BFpacketcapture_impl(capture, nullptr, nullptr, ring, nsrc, buffer_ntime, slot_ntime), 
		  _type_log((std::string(capture->get_name())+"/type").c_str()),
		  _chan_log((std::string(capture->get_name())+"/chans").c_str()),
		  _sequence_callback(sequence_callback->get_tbn()),
          _decim(0) {
		_decoder = new TBNDecoder(nsrc, src0);
		_processor = new TBNProcessor();
		_type_log.update("type : %s\n", "tbn");
	}
};
                                                    
class BFpacketcapture_drx_impl : public BFpacketcapture_impl {
	ProcLog          _type_log;
	ProcLog          _chan_log;
	
	BFpacketcapture_drx_sequence_callback _sequence_callback;
	
	BFoffset _time_tag;
    uint16_t _decim;
	int      _chan1;
	
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
	    return (   (pkt->tuning  != _chan0)
	            || (pkt->tuning1 != _chan1)
                || (pkt->decimation != _decim) );
	}
	void on_sequence_changed(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size) {
	    *seq0 = _seq;// + _nseq_per_buf*_bufs.size();
	    *time_tag = pkt->time_tag;
        _time_tag     = pkt->time_tag;
        _chan0        = pkt->tuning;
        _chan1        = pkt->tuning1;
        if( _nsrc == 2 ) {
            _chan0        = std::max(_chan0, _chan1);
            _chan1        = 0;
        }
        _decim        = pkt->decimation;
        _payload_size = pkt->payload_size;
        
	    if( _sequence_callback ) {
	        int status = (*_sequence_callback)(*seq0,
	                                        *time_tag,
                                            _decim,
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
        
        _chan_log.update() << "chan0        : " << _chan0 << "\n"
					       << "chan1        : " << _chan1 << "\n"
					       << "payload_size : " << _payload_size << "\n";
    }
public:
	inline BFpacketcapture_drx_impl(PacketCaptureThread* capture,
	                                BFring               ring,
	                                int                  nsrc,
	                                int                  src0,
	                                int                  buffer_ntime,
	                                int                  slot_ntime,
	                                BFpacketcapture_callback sequence_callback)
		: BFpacketcapture_impl(capture, nullptr, nullptr, ring, nsrc, buffer_ntime, slot_ntime), 
		  _type_log((std::string(capture->get_name())+"/type").c_str()),
		  _chan_log((std::string(capture->get_name())+"/chans").c_str()),
		  _sequence_callback(sequence_callback->get_drx()), 
		  _decim(0), _chan1(0) {
		_decoder = new DRXDecoder(nsrc, src0);
		_processor = new DRXProcessor();
		_type_log.update("type : %s\n", "drx");
	}
};

BFstatus BFpacketcapture_create(BFpacketcapture* obj,
                                const char*      format,
                                int              fd,
                                BFring           ring,
                                BFsize           nsrc,
                                BFsize           src0,
                                BFsize           max_payload_size,
                                BFsize           buffer_ntime,
                                BFsize           slot_ntime,
                                BFpacketcapture_callback sequence_callback,
                                int              core,
                                BFiomethod       backend) {
    BF_ASSERT(obj, BF_STATUS_INVALID_POINTER);
    
    if( std::string(format).substr(0, 5) == std::string("chips") ) {
        if( backend == BF_IO_DISK ) {
            // Need to know how much to read at a time
            int nchan = std::atoi((std::string(format).substr(6, std::string(format).length())).c_str());
            max_payload_size = sizeof(chips_hdr_type) + 32*nchan;
        }
    } else if( std::string(format).substr(0, 6) == std::string("ibeam2") ) {
        if( backend == BF_IO_DISK ) {
            // Need to know how much to read at a time
            int nchan = std::atoi((std::string(format).substr(7, std::string(format).length())).c_str());
            max_payload_size = sizeof(ibeam_hdr_type) + 64*2*2*nchan;
        }
    } else if( std::string(format).substr(0, 6) == std::string("ibeam3") ) {
        if( backend == BF_IO_DISK ) {
            // Need to know how much to read at a time
            int nchan = std::atoi((std::string(format).substr(7, std::string(format).length())).c_str());
            max_payload_size = sizeof(ibeam_hdr_type) + 64*2*3*nchan;
        }
    } else if( std::string(format).substr(0, 6) == std::string("ibeam4") ) {
        if( backend == BF_IO_DISK ) {
            // Need to know how much to read at a time
            int nchan = std::atoi((std::string(format).substr(7, std::string(format).length())).c_str());
            max_payload_size = sizeof(ibeam_hdr_type) + 64*2*4*nchan;
        }
    } else if(std::string(format).substr(0, 3) == std::string("cor") ) {
        if( backend == BF_IO_DISK ) {
            // Need to know how much to read at a time
            int nchan = std::atoi((std::string(format).substr(4, std::string(format).length())).c_str());
            max_payload_size = sizeof(cor_hdr_type) + (8*4*nchan);
        }
    } else if(std::string(format).substr(0, 4) == std::string("vdif") ) {
        if( backend == BF_IO_DISK ) {
            // Need to know how much to read at a time
            int nchan = std::atoi((std::string(format).substr(5, std::string(format).length())).c_str());
            max_payload_size = nchan;
        }
    } else if( format == std::string("tbn") ) {
        max_payload_size = TBN_FRAME_SIZE;
    } else if( format == std::string("drx") ) {
        max_payload_size = DRX_FRAME_SIZE;
    }
    
    PacketCaptureMethod* method;
    if( backend == BF_IO_DISK ) {
        method = new DiskPacketReader(fd, max_payload_size);
    } else if( backend == BF_IO_UDP ) {
        method = new UDPPacketReceiver(fd, max_payload_size);
    } else if( backend == BF_IO_SNIFFER ) {
        method = new UDPPacketSniffer(fd, max_payload_size);
#if BF_VERBS_ENABLED
    } else if( backend == BF_IO_VERBS ) {
        method = new UDPVerbsReceiver(fd, max_payload_size, core);
#endif
    } else {
        return BF_STATUS_UNSUPPORTED;
    }
    PacketCaptureThread* capture = new PacketCaptureThread(method, nsrc, core);
    
    if( std::string(format).substr(0, 5) == std::string("chips") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketcapture_chips_impl(capture, ring, nsrc, src0,
                                                                 buffer_ntime, slot_ntime,
                                                                 sequence_callback),
                           *obj = 0);
    } else if( std::string(format).substr(0, 6) == std::string("ibeam2") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketcapture_ibeam_impl<2>(capture, ring, nsrc, src0,
                                                                    buffer_ntime, slot_ntime,
                                                                    sequence_callback),
                           *obj = 0);
    } else if( std::string(format).substr(0, 6) == std::string("ibeam3") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketcapture_ibeam_impl<3>(capture, ring, nsrc, src0,
                                                                    buffer_ntime, slot_ntime,
                                                                    sequence_callback),
                           *obj = 0);
    } else if( std::string(format).substr(0, 6) == std::string("ibeam4") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketcapture_ibeam_impl<4>(capture, ring, nsrc, src0,
                                                                    buffer_ntime, slot_ntime,
                                                                    sequence_callback),
                           *obj = 0);
    } else if( std::string(format).substr(0, 3) == std::string("cor") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketcapture_cor_impl(capture, ring, nsrc, src0,
                                                               buffer_ntime, slot_ntime,
                                                               sequence_callback),
                           *obj = 0);
    } else if( std::string(format).substr(0, 4) == std::string("vdif") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketcapture_vdif_impl(capture, ring, nsrc, src0,
                                                                buffer_ntime, slot_ntime,
                                                                sequence_callback),
                           *obj = 0);
    } else if( format == std::string("tbn") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketcapture_tbn_impl(capture, ring, nsrc, src0,
                                                               buffer_ntime, slot_ntime,
                                                               sequence_callback),
                           *obj = 0);
    } else if( format == std::string("drx") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketcapture_drx_impl(capture, ring, nsrc, src0,
                                                               buffer_ntime, slot_ntime,
                                                               sequence_callback),
                           *obj = 0);
    } else {
        return BF_STATUS_UNSUPPORTED;
    }
}
