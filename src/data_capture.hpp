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
#include <bifrost/data_capture.h>
#include <bifrost/affinity.h>
#include <bifrost/Ring.hpp>
using bifrost::ring::RingWrapper;
using bifrost::ring::RingWriter;
using bifrost::ring::WriteSpan;
using bifrost::ring::WriteSequence;
#include "proclog.hpp"
#include "formats/formats.hpp"

#include <cstdlib>      // For posix_memalign
#include <cstring>      // For memcpy, memset
#include <cstdint>

#include <queue>
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
	inline size_t size() const                 { return _size; }
	inline T      & operator[](size_t i)       { return _buf[i]; }
	inline T const& operator[](size_t i) const { return _buf[i]; }
};

#if BF_HWLOC_ENABLED
#include <hwloc.h>
class HardwareLocality {
	hwloc_topology_t _topo;
	HardwareLocality(HardwareLocality const&);
	HardwareLocality& operator=(HardwareLocality const&);
public:
	HardwareLocality() {
		hwloc_topology_init(&_topo);
		hwloc_topology_load(_topo);
	}
	~HardwareLocality() {
		hwloc_topology_destroy(_topo);
	}
	int bind_memory_to_core(int core);
};
#endif // BF_HWLOC_ENABLED

class BoundThread {
#if BF_HWLOC_ENABLED
	HardwareLocality _hwloc;
#endif
public:
	BoundThread(int core) {
		bfAffinitySetCore(core);
#if BF_HWLOC_ENABLED
		assert(_hwloc.bind_memory_to_core(core) == 0);
#endif
	}
};

class DataCaptureMethod {
protected:
    int                    _fd;
	AlignedBuffer<uint8_t> _buf;
public:
	DataCaptureMethod(int fd, size_t pkt_size_max=9000)
	: _fd(fd), _buf(pkt_size_max)
	{}
	virtual inline int recv_packet(uint8_t** pkt_ptr, int flags=0) {
	    return 0;
	}
};

struct PacketStats {
	size_t ninvalid;
	size_t ninvalid_bytes;
	size_t nlate;
	size_t nlate_bytes;
	size_t nvalid;
	size_t nvalid_bytes;
};

class DataCaptureThread : public BoundThread {
protected:
    DataCaptureMethod        _mthd;
    PacketStats              _stats;
	std::vector<PacketStats> _src_stats;
	bool                     _have_pkt;
	PacketDesc               _pkt;
public:
	enum {
		CAPTURE_SUCCESS     = 1 << 0,
		CAPTURE_TIMEOUT     = 1 << 1,
		CAPTURE_INTERRUPTED = 1 << 2,
		CAPTURE_ERROR       = 1 << 3
	};
	DataCaptureThread(int fd, int nsrc, int core=0, size_t pkt_size_max=9000)
		: BoundThread(core), _mthd(fd, pkt_size_max), _src_stats(nsrc),
		  _have_pkt(false) {
		this->reset_stats();
	}
	virtual int run(uint64_t         seq_beg,
	                uint64_t         nseq_per_obuf,
	                int              nbuf,
	                uint8_t*         obufs[],
	                size_t*          ngood_bytes[],
	                size_t*          src_ngood_bytes[],
	                PacketDecoder*   decode,
	                PacketProcessor* process) {
        return CAPTURE_ERROR;
    }
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

class BFdatacapture_impl {
protected:
    std::string        _name;
	DataCaptureThread  _capture;
	PacketDecoder      _decoder;
	PacketProcessor    _processor;
	ProcLog            _bind_log;
	ProcLog            _out_log;
	ProcLog            _size_log;
	ProcLog            _stat_log;
	ProcLog            _perf_log;
	pid_t              _pid;
	
	std::chrono::high_resolution_clock::time_point _t0;
	std::chrono::high_resolution_clock::time_point _t1;
	std::chrono::high_resolution_clock::time_point _t2;
	std::chrono::duration<double> _process_time;
	std::chrono::duration<double> _reserve_time;
	
	int      _nsrc;
	int      _nseq_per_buf;
	int      _slot_ntime;
	BFoffset _seq;
	int      _chan0;
	int      _nchan;
	int      _payload_size;
	bool     _active;
	
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
				_processor.blank_out_source(data, src, _nsrc,
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
	virtual inline bool has_sequence_changed(const PacketDesc* pkt) {
	    return false;
	}
	virtual void on_sequence_changed(const PacketDesc* pkt, BFoffset* seq0, BFoffset* time_tag, const void** hdr, size_t* hdr_size) {}
public:
	inline BFdatacapture_impl(std::string name,
				              int         fd,
	                          BFring      ring,
            	              int         nsrc,
	                          int         src0,
	                          int         max_payload_size,
	                          int         buffer_ntime,
	                          int         slot_ntime,
	                          int         core)
		: _name(name), _capture(fd, nsrc, core), _decoder(nsrc, src0), _processor(),
		  _bind_log(_name+"/bind"),
		  _out_log(_name+"/out"),
		  _size_log(_name+"/sizes"),
		  _stat_log(_name+"/stats"),
		  _perf_log(_name+"/perf"), 
		  _nsrc(nsrc), _nseq_per_buf(buffer_ntime), _slot_ntime(slot_ntime),
		  _seq(), _chan0(), _nchan(), _active(false),
		  _ring(ring), _oring(_ring),
		  // TODO: Add reset method for stats
		  _ngood_bytes(0), _nmissing_bytes(0) {}
    virtual ~BFdatacapture_impl() {}
	inline void flush() {
		while( _bufs.size() ) {
			this->commit_buf();
		}
		if( _sequence ) {
			this->end_sequence();
		}
	}
	inline void end_writing() {
		this->flush();
		_oring.close();
	}
	BFdatacapture_status recv();
};
