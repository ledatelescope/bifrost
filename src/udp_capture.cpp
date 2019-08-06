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
#include <bifrost/udp_capture.h>
#include <bifrost/affinity.h>
#include <bifrost/Ring.hpp>
using bifrost::ring::RingWrapper;
using bifrost::ring::RingWriter;
using bifrost::ring::WriteSpan;
using bifrost::ring::WriteSequence;
#include "proclog.hpp"

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

//#include <immintrin.h> // SSE

// TODO: The VMA API is returning unaligned buffers, which prevents use of SSE
#ifndef BF_VMA_ENABLED
#define BF_VMA_ENABLED 0
//#define BF_VMA_ENABLED 1
#endif

#ifndef BF_HWLOC_ENABLED
#define BF_HWLOC_ENABLED 0
//#define BF_HWLOC_ENABLED 1
#endif

#define BF_UNPACK_FACTOR 1

enum {
	JUMBO_FRAME_SIZE = 9000
};

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
	int bind_memory_to_core(int core) {
		int core_depth = hwloc_get_type_or_below_depth(_topo, HWLOC_OBJ_CORE);
		int ncore      = hwloc_get_nbobjs_by_depth(_topo, core_depth);
		int ret = 0;
		if( 0 <= core && core < ncore ) {
			hwloc_obj_t    obj    = hwloc_get_obj_by_depth(_topo, core_depth, core);
			hwloc_cpuset_t cpuset = hwloc_bitmap_dup(obj->cpuset);
			hwloc_bitmap_singlify(cpuset); // Avoid hyper-threads
			hwloc_membind_policy_t policy = HWLOC_MEMBIND_BIND;
			hwloc_membind_flags_t  flags  = HWLOC_MEMBIND_THREAD;
			ret = hwloc_set_membind(_topo, cpuset, policy, flags);
			hwloc_bitmap_free(cpuset);
		}
		return ret;
	}
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

class UDPPacketReceiver {
	int                    _fd;
	AlignedBuffer<uint8_t> _buf;
#if BF_VMA_ENABLED
	VMAReceiver            _vma;
#endif
public:
	UDPPacketReceiver(int fd, size_t pkt_size_max=JUMBO_FRAME_SIZE)
		: _fd(fd), _buf(pkt_size_max)
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

struct PacketDesc {
	uint64_t       seq;
	int            nsrc;
	int            src;
	int            nchan;
	int            chan0;
	int            payload_size;
	const uint8_t* payload_ptr;
};
struct PacketStats {
	size_t ninvalid;
	size_t ninvalid_bytes;
	size_t nlate;
	size_t nlate_bytes;
	size_t nvalid;
	size_t nvalid_bytes;
};

class UDPCaptureThread : public BoundThread {
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
		: BoundThread(core), _udp(fd, pkt_size_max), _src_stats(nsrc),
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
	inline const PacketDesc* get_last_packet() const {
		return _have_pkt ? &_pkt : NULL;
	}
	inline const PacketStats* get_stats() const { return &_stats; }
	inline const PacketStats* get_stats(int src) const { return &_src_stats[src]; }
	inline void reset_stats() {
		::memset(&_stats, 0, sizeof(_stats));
		::memset(&_src_stats[0], 0, _src_stats.size()*sizeof(PacketStats));
	}
};

#pragma pack(1)
struct chips_hdr_type {
	uint8_t  roach;    // Note: 1-based
	uint8_t  gbe;      // (AKA tuning)
	uint8_t  nchan;    // 109
	uint8_t  nsubband; // 11
	uint8_t  subband;  // 0-11
	uint8_t  nroach;   // 16
	// Note: Big endian
	uint16_t chan0;    // First chan in packet
	uint64_t seq;      // Note: 1-based
};

class CHIPSDecoder {
	// TODO: See if can remove these once firmware supports per-GbE nroach
	int _nsrc;
	int _src0;
	inline bool valid_packet(const PacketDesc* pkt) const {
		return (pkt->seq   >= 0 &&
		        pkt->src   >= 0 && pkt->src < _nsrc &&
		        pkt->chan0 >= 0);
	}
public:
	CHIPSDecoder(int nsrc, int src0) : _nsrc(nsrc), _src0(src0) {}
	inline bool operator()(const uint8_t* pkt_ptr,
	                       int            pkt_size,
	                       PacketDesc*    pkt) const {
		if( pkt_size < (int)sizeof(chips_hdr_type) ) {
			return false;
		}
		const chips_hdr_type* pkt_hdr  = (chips_hdr_type*)pkt_ptr;
		const uint8_t*        pkt_pld  = pkt_ptr  + sizeof(chips_hdr_type);
		int                   pld_size = pkt_size - sizeof(chips_hdr_type);
		pkt->seq   = be64toh(pkt_hdr->seq)  - 1;
		//pkt->nsrc  =         pkt_hdr->nroach;
		pkt->nsrc  =         _nsrc;
		pkt->src   =        (pkt_hdr->roach - 1) - _src0;
		pkt->nchan =         pkt_hdr->nchan;
		pkt->chan0 =   ntohs(pkt_hdr->chan0);
		pkt->payload_size = pld_size;
		pkt->payload_ptr  = pkt_pld;
		return this->valid_packet(pkt);
	}
};

struct __attribute__((aligned(32))) aligned256_type {
	uint8_t data[32];
};
struct __attribute__((aligned(64))) aligned512_type {
	uint8_t data[64];
};

class CHIPSProcessor8bit {
public:
	inline void operator()(const PacketDesc* pkt,
	                       uint64_t          seq0,
	                       uint64_t          nseq_per_obuf,
	                       int               nbuf,
	                       uint8_t*          obufs[],
	                       size_t            ngood_bytes[],
	                       size_t*           src_ngood_bytes[]) {
		enum {
			PKT_NINPUT = 32,
			PKT_NBIT   = 4
		};
		int    obuf_idx = ((pkt->seq - seq0 >= 1*nseq_per_obuf) +
		                   (pkt->seq - seq0 >= 2*nseq_per_obuf));
		size_t obuf_seq0 = seq0 + obuf_idx*nseq_per_obuf;
		size_t nbyte = pkt->payload_size * BF_UNPACK_FACTOR;
		ngood_bytes[obuf_idx]               += nbyte;
		src_ngood_bytes[obuf_idx][pkt->src] += nbyte;
		// **CHANGED RECENTLY
		int payload_size = pkt->payload_size;//pkt->nchan*(PKT_NINPUT*2*PKT_NBIT/8);
		
		size_t obuf_offset = (pkt->seq-obuf_seq0)*pkt->nsrc*payload_size;
		typedef aligned256_type itype;
		typedef aligned256_type otype;
		
		obuf_offset *= BF_UNPACK_FACTOR;
		
		// Note: Using these SSE types allows the compiler to use SSE instructions
		//         However, they require aligned memory (otherwise segfault)
		itype const* __restrict__ in  = (itype const*)pkt->payload_ptr;
		otype*       __restrict__ out = (otype*      )&obufs[obuf_idx][obuf_offset];
		
		int chan = 0;
		//cout << pkt->src << ", " << pkt->nsrc << endl;
		//cout << pkt->nchan << endl;
		/*
		  // HACK TESTING disabled
		for( ; chan<pkt->nchan/4*4; chan+=4 ) {
			__m128i tmp0 = ((__m128i*)&in[chan])[0];
			__m128i tmp1 = ((__m128i*)&in[chan])[1];
			__m128i tmp2 = ((__m128i*)&in[chan+1])[0];
			__m128i tmp3 = ((__m128i*)&in[chan+1])[1];
			__m128i tmp4 = ((__m128i*)&in[chan+2])[0];
			__m128i tmp5 = ((__m128i*)&in[chan+2])[1];
			__m128i tmp6 = ((__m128i*)&in[chan+3])[0];
			__m128i tmp7 = ((__m128i*)&in[chan+3])[1];
			_mm_store_si128(&((__m128i*)&out[pkt->src + pkt->nsrc*chan])[0], tmp0);
			_mm_store_si128(&((__m128i*)&out[pkt->src + pkt->nsrc*chan])[1], tmp1);
			_mm_store_si128(&((__m128i*)&out[pkt->src + pkt->nsrc*(chan+1)])[0], tmp2);
			_mm_store_si128(&((__m128i*)&out[pkt->src + pkt->nsrc*(chan+1)])[1], tmp3);
			_mm_store_si128(&((__m128i*)&out[pkt->src + pkt->nsrc*(chan+2)])[0], tmp4);
			_mm_store_si128(&((__m128i*)&out[pkt->src + pkt->nsrc*(chan+2)])[1], tmp5);
			_mm_store_si128(&((__m128i*)&out[pkt->src + pkt->nsrc*(chan+3)])[0], tmp6);
			_mm_store_si128(&((__m128i*)&out[pkt->src + pkt->nsrc*(chan+3)])[1], tmp7);
		}
		*/
		//for( ; chan<pkt->nchan; ++chan ) {
		/*
		for( ; chan<10; ++chan ) { // HACK TESTING
			__m128i tmp0 = ((__m128i*)&in[chan])[0];
			__m128i tmp1 = ((__m128i*)&in[chan])[1];
			_mm_store_si128(&((__m128i*)&out[pkt->src + pkt->nsrc*chan])[0], tmp0);
			_mm_store_si128(&((__m128i*)&out[pkt->src + pkt->nsrc*chan])[1], tmp1);
		}
		*/
		//if( pkt->src < 8 ) { // HACK TESTING
		//for( ; chan<32; ++chan ) { // HACK TESTING
		for( ; chan<pkt->nchan; ++chan ) { // HACK TESTING
			out[pkt->src + pkt->nsrc*chan] = in[chan];
			//::memset(
		}
		//}
	}
	inline void blank_out_source(uint8_t* data,
	                             int      src,
	                             int      nsrc,
	                             int      nchan,
	                             int      nseq) {
		typedef aligned256_type otype;
		otype* __restrict__ aligned_data = (otype*)data;
		for( int t=0; t<nseq; ++t ) {
			for( int c=0; c<nchan; ++c ) {
				::memset(&aligned_data[src + nsrc*(c + nchan*t)],
				         0, sizeof(otype));
			}
		}
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

class BFudpcapture_impl {
	UDPCaptureThread   _capture;
	CHIPSDecoder       _decoder;
	CHIPSProcessor8bit _processor;
	ProcLog            _type_log;
	ProcLog            _bind_log;
	ProcLog            _out_log;
	ProcLog            _size_log;
	ProcLog            _chan_log;
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
	BFudpcapture_sequence_callback _sequence_callback;
	
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
	inline void begin_sequence() {
		BFoffset    seq0 = _seq;// + _nseq_per_buf*_bufs.size();
		const void* hdr;
		size_t      hdr_size;
		BFoffset    time_tag;
		if( _sequence_callback ) {
			int status = (*_sequence_callback)(seq0,
			                                 _chan0,
			                                 _nchan,
			                                 _nsrc,
			                                 &time_tag,
			                                 &hdr,
			                                 &hdr_size);
			if( status != 0 ) {
				// TODO: What to do here? Needed?
				throw std::runtime_error("BAD HEADER CALLBACK STATUS");
			}
		} else {
			// Simple default for easy testing
			time_tag = seq0;
			hdr      = NULL;
			hdr_size = 0;
		}
		const char* name     = "";
		int         nringlet = 1;
		_sequence.reset(new WriteSequence(_oring, name, time_tag,
		                                  hdr_size, hdr, nringlet));
	}
	inline void end_sequence() {
		_sequence.reset(); // Note: This is releasing the shared_ptr
	}
public:
	inline BFudpcapture_impl(int    fd,
	           BFring ring,
	           int    nsrc,
	           int    src0,
	           int    max_payload_size,
	           int    buffer_ntime,
	           int    slot_ntime,
	           BFudpcapture_sequence_callback sequence_callback,
	           int    core)
		: _capture(fd, nsrc, core), _decoder(nsrc, src0), _processor(),
		  _type_log("udp_capture/type"),
		  _bind_log("udp_capture/bind"),
		  _out_log("udp_capture/out"),
		  _size_log("udp_capture/sizes"),
		  _chan_log("udp_capture/chans"),
		  _stat_log("udp_capture/stats"),
		  _perf_log("udp_capture/perf"), 
		  _nsrc(nsrc), _nseq_per_buf(buffer_ntime), _slot_ntime(slot_ntime),
		  _seq(), _chan0(), _nchan(), _active(false),
		  _sequence_callback(sequence_callback),
		  _ring(ring), _oring(_ring),
		  // TODO: Add reset method for stats
		  _ngood_bytes(0), _nmissing_bytes(0) {
		size_t contig_span  = this->bufsize(max_payload_size);
		// Note: 2 write bufs may be open for writing at one time
		size_t total_span   = contig_span * 4;
		size_t nringlet_max = 1;
		_ring.resize(contig_span, total_span, nringlet_max);
		_type_log.update("type : %s", "chips");
		_bind_log.update("ncore : %i\n"
		                 "core0 : %i\n", 
		                 1, core);
		_out_log.update("nring : %i\n"
		                "ring0 : %s\n", 
		                1, _ring.name());
		_size_log.update("nsrc         : %i\n"
		                 "nseq_per_buf : %i\n"
		                 "slot_ntime   : %i\n",
		                 _nsrc, _nseq_per_buf, _slot_ntime);
	}
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
	BFudpcapture_status recv() {
		_t0 = std::chrono::high_resolution_clock::now();
		
		uint8_t* buf_ptrs[2];
		// Minor HACK to access the buffers in a 2-element queue
		buf_ptrs[0] = _bufs.size() > 0 ? (uint8_t*)_bufs.front()->data() : NULL;
		buf_ptrs[1] = _bufs.size() > 1 ? (uint8_t*)_bufs.back()->data()  : NULL;
		
		size_t* ngood_bytes_ptrs[2];
		ngood_bytes_ptrs[0] = _buf_ngood_bytes.size() > 0 ? &_buf_ngood_bytes.front() : NULL;
		ngood_bytes_ptrs[1] = _buf_ngood_bytes.size() > 1 ? &_buf_ngood_bytes.back()  : NULL;
		
		size_t* src_ngood_bytes_ptrs[2];
		src_ngood_bytes_ptrs[0] = _buf_src_ngood_bytes.size() > 0 ? &_buf_src_ngood_bytes.front()[0] : NULL;
		src_ngood_bytes_ptrs[1] = _buf_src_ngood_bytes.size() > 1 ? &_buf_src_ngood_bytes.back()[0]  : NULL;
		
		int state = _capture.run(_seq,
		                         _nseq_per_buf,
		                         _bufs.size(),
		                         buf_ptrs,
		                         ngood_bytes_ptrs,
		                         src_ngood_bytes_ptrs,
		                         &_decoder,
		                         &_processor);
		if( state & UDPCaptureThread::CAPTURE_ERROR ) {
			return BF_CAPTURE_ERROR;
		} else if( state & UDPCaptureThread::CAPTURE_INTERRUPTED ) {
			return BF_CAPTURE_INTERRUPTED;
		}
		const PacketStats* stats = _capture.get_stats();
		_stat_log.update() << "ngood_bytes    : " << _ngood_bytes << "\n"
		                   << "nmissing_bytes : " << _nmissing_bytes << "\n"
		                   << "ninvalid       : " << stats->ninvalid << "\n"
		                   << "ninvalid_bytes : " << stats->ninvalid_bytes << "\n"
		                   << "nlate          : " << stats->nlate << "\n"
		                   << "nlate_bytes    : " << stats->nlate_bytes << "\n"
		                   << "nvalid         : " << stats->nvalid << "\n"
		                   << "nvalid_bytes   : " << stats->nvalid_bytes << "\n";
		
		_t1 = std::chrono::high_resolution_clock::now();
		
		BFudpcapture_status ret;
		bool was_active = _active;
		_active = state & UDPCaptureThread::CAPTURE_SUCCESS;
		if( _active ) {
			const PacketDesc* pkt = _capture.get_last_packet();
			if( pkt ) {
				//cout << "Latest nchan, chan0 = " << pkt->nchan << ", " << pkt->chan0 << endl;
			}
			else {
				//cout << "No latest packet" << endl;
			}
			if( !was_active ) {
				//cout << "Beginning of sequence, first pkt seq = " << pkt->seq << endl;
				// TODO: Might be safer to round to nearest here, but the current firmware
				//         always starts things ~3 seq's before the 1sec boundary anyway.
				//seq = round_up(pkt->seq, _slot_ntime);
				//*_seq          = round_nearest(pkt->seq, _slot_ntime);
				_seq          = round_up(pkt->seq, _slot_ntime);
				_chan0        = pkt->chan0;
				_nchan        = pkt->nchan;
				_payload_size = pkt->payload_size;
				_chan_log.update() << "chan0        : " << _chan0 << "\n"
				                   << "nchan        : " << _nchan << "\n"
				                   << "payload_size : " << _payload_size << "\n";
				this->begin_sequence();
				ret = BF_CAPTURE_STARTED;
			} else {
				//cout << "Continuing data, seq = " << seq << endl;
				if( pkt->chan0 != _chan0 ||
				    pkt->nchan != _nchan ) {
					_chan0 = pkt->chan0;
					_nchan = pkt->nchan;
					_payload_size = pkt->payload_size;
					_chan_log.update() << "chan0        : " << _chan0 << "\n"
					                   << "nchan        : " << _nchan << "\n"
					                   << "payload_size : " << _payload_size << "\n";
					this->end_sequence();
					this->begin_sequence();
					ret = BF_CAPTURE_CHANGED;
				} else {
					ret = BF_CAPTURE_CONTINUED;
				}
			}
			if( _bufs.size() == 2 ) {
				this->commit_buf();
			}
			this->reserve_buf();
		} else {
			
			if( was_active ) {
				this->flush();
				ret = BF_CAPTURE_ENDED;
			} else {
				ret = BF_CAPTURE_NO_DATA;
			}
		}
		
		_t2 = std::chrono::high_resolution_clock::now();
		_process_time = std::chrono::duration_cast<std::chrono::duration<double>>(_t1-_t0);
		_reserve_time = std::chrono::duration_cast<std::chrono::duration<double>>(_t2-_t1);
		_perf_log.update() << "acquire_time : " << -1.0 << "\n"
		                   << "process_time : " << _process_time.count() << "\n"
		                   << "reserve_time : " << _reserve_time.count() << "\n";
		
		return ret;
	}
};

BFstatus bfUdpCaptureCreate(BFudpcapture* obj,
                            const char*   format,
                            int           fd,
                            BFring        ring,
                            BFsize        nsrc,
                            BFsize        src0,
                            BFsize        max_payload_size,
                            BFsize        buffer_ntime,
                            BFsize        slot_ntime,
                            BFudpcapture_sequence_callback sequence_callback,
                            int           core) {
	BF_ASSERT(obj, BF_STATUS_INVALID_POINTER);
	if( format == std::string("chips") ) {
		BF_TRY_RETURN_ELSE(*obj = new BFudpcapture_impl(fd, ring, nsrc, src0, max_payload_size,
		                                                buffer_ntime, slot_ntime,
		                                                sequence_callback, core),
		                   *obj = 0);
	} else {
		return BF_STATUS_UNSUPPORTED;
	}
}
BFstatus bfUdpCaptureDestroy(BFudpcapture obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	delete obj;
	return BF_STATUS_SUCCESS;
}
BFstatus bfUdpCaptureRecv(BFudpcapture obj, BFudpcapture_status* result) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*result = obj->recv(),
	                   *result = BF_CAPTURE_ERROR);
}
BFstatus bfUdpCaptureFlush(BFudpcapture obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(obj->flush());
}
BFstatus bfUdpCaptureEnd(BFudpcapture obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(obj->end_writing());
}
