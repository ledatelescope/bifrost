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
#include <bifrost/drx_reader.h>
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

#ifndef BF_HWLOC_ENABLED
#define BF_HWLOC_ENABLED 0
//#define BF_HWLOC_ENABLED 1
#endif

enum {
	DRX_FRAME_SIZE   = 4128
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
	: _buf(0), _size(other._size), _alignment(other._alignment) {
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
	inline void swap(AlignedBuffer const& other) {
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

class DRXReaderReader {
	int                    _fd;
	AlignedBuffer<uint8_t> _buf;
public:
	DRXReaderReader(int fd, size_t frm_size_max=DRX_FRAME_SIZE)
	: _fd(fd), _buf(frm_size_max)
	{}
	inline int read_frame(uint8_t** frm_ptr, int flags=0) {
		*frm_ptr = &_buf[0];
		return ::read(_fd, &_buf[0], _buf.size());
	}
};


struct FrameDesc {
	uint32_t       sync;
	uint64_t       time_tag;
	uint64_t       seq;
	int            nsrc;
	int            src;
	int            tuning;
	int            decimation;
	int            valid_mode;
	int            payload_size;
	const uint8_t* payload_ptr;
};
struct FrameStats {
	size_t ninvalid;
	size_t ninvalid_bytes;
	size_t nlate;
	size_t nlate_bytes;
	size_t nvalid;
	size_t nvalid_bytes;
};

class DRXReaderThread : public BoundThread {
	DRXReaderReader _disk;
	FrameStats       _stats;
	std::vector<FrameStats> _src_stats;
	bool             _have_frm;
	FrameDesc        _frm;
public:
	enum {
		READ_SUCCESS     = 1 << 0,
		READ_TIMEOUT     = 1 << 1,
		READ_INTERRUPTED = 1 << 2,
		READ_ERROR       = 1 << 3
	};
	DRXReaderThread(int fd, int nsrc, int core=0, size_t frm_size_max=9000)
	: BoundThread(core), _disk(fd, frm_size_max), _src_stats(nsrc),
	_have_frm(false) {
		this->reset_stats();
	}
	// Reads, decodes and unpacks frames into the provided buffers
	// Note: Read continues until the first frame that belongs
	//         beyond the end of the provided buffers. This frame is
	//         saved, accessible via get_last_frame(), and will be
	//         processed on the next call to run() if possible.
	template<class FrameDecoder, class FrameProcessor>
	int run(uint64_t         seq_beg,
	        uint64_t         nseq_per_obuf,
	        int              nbuf,
	        uint8_t*         obufs[],
	        size_t*          ngood_bytes[],
	        size_t*          src_ngood_bytes[],
	        FrameDecoder*    decode,
	        FrameProcessor*  process) {
		uint64_t seq_end = seq_beg + nbuf*nseq_per_obuf;
		size_t local_ngood_bytes[2] = {0, 0};
		int ret;
		while( true ) {
			if( !_have_frm ) {
				uint8_t* frm_ptr;
				int frm_size = _disk.read_frame(&frm_ptr);
				if( frm_size <= 0 ) {
					if( errno == EAGAIN || errno == EWOULDBLOCK ) {
						ret = READ_TIMEOUT; // Timed out
					} else if( errno == EINTR ) {
						ret = READ_INTERRUPTED; // Interrupted by signal
					} else {
						ret = READ_ERROR; // Socket error
					}
					break;
				}
				if( !(*decode)(frm_ptr, frm_size, &_frm) ) {
					++_stats.ninvalid;
					_stats.ninvalid_bytes += frm_size;
					continue;
				}
				_have_frm = true;
			}
			if( greater_equal(_frm.seq, seq_end) ) {
				// Reached the end of this processing gulp, so leave this
				//   frame unprocessed and return.
				ret = READ_SUCCESS;
				break;
			}
			_have_frm = false;
			if( less_than(_frm.seq, seq_beg) ) {
				++_stats.nlate;
				_stats.nlate_bytes += _frm.payload_size;
				++_src_stats[_frm.src].nlate;
				_src_stats[_frm.src].nlate_bytes += _frm.payload_size;
				continue;
			}
			++_stats.nvalid;
			_stats.nvalid_bytes += _frm.payload_size;
			++_src_stats[_frm.src].nvalid;
			_src_stats[_frm.src].nvalid_bytes += _frm.payload_size;
			// HACK TODO: src_ngood_bytes should be accumulated locally and
			//              then atomically updated, like ngood_bytes. The
			//              current way is not thread-safe.
			(*process)(&_frm, seq_beg, nseq_per_obuf, nbuf, obufs,
					 local_ngood_bytes, /*local_*/src_ngood_bytes);
		}
		if( nbuf > 0 ) { atomic_add_and_fetch(ngood_bytes[0], local_ngood_bytes[0]); }
		if( nbuf > 1 ) { atomic_add_and_fetch(ngood_bytes[1], local_ngood_bytes[1]); }
		return ret;
	}
	inline const FrameDesc* get_last_frame() const {
		return _have_frm ? &_frm : NULL;
	}
	inline void reset_last_frame() {
		_have_frm = false;
	}
	inline const FrameStats* get_stats() const { return &_stats; }
	inline const FrameStats* get_stats(int src) const { return &_src_stats[src]; }
	inline void reset_stats() {
		::memset(&_stats, 0, sizeof(_stats));
		::memset(&_src_stats[0], 0, _src_stats.size()*sizeof(FrameStats));
	}
};

#pragma pack(1)
struct drx_hdr_type {
	uint32_t sync_word;
	uint32_t frame_count_word;
	uint32_t seconds_count;
	uint16_t decimation;
	uint16_t time_offset;
	uint64_t time_tag;
	uint32_t tuning_word;
	uint32_t flags;
};

class DRXDecoder {
protected:
	int _nsrc;
	int _src0;
	inline bool valid_frame(const FrameDesc* frm) const {
		return (frm->sync     == 0x5CDEC0DE &&
		frm->src      >= 0 &&
		frm->src      <  _nsrc &&
		frm->time_tag >= 0 &&
		frm->tuning   >= 0);
	}
public:
	DRXDecoder(int nsrc, int src0) : _nsrc(nsrc), _src0(src0) {}
	inline bool operator()(const uint8_t* frm_ptr,
	                       int            frm_size,
	                       FrameDesc*     frm) const {
		if( frm_size < (int)sizeof(drx_hdr_type) ) {
			return false;
		}
		const drx_hdr_type* frm_hdr  = (drx_hdr_type*)frm_ptr;
		const uint8_t*      frm_pld  = frm_ptr  + sizeof(drx_hdr_type);
		int                 pld_size = frm_size - sizeof(drx_hdr_type);
		int frm_id   = frm_hdr->frame_count_word & 0xFF;
		//uint8_t frm_beam = (frm_id & 0x7) - 1;
		int frm_tune = ((frm_id >> 3) & 0x7) - 1;
		int frm_pol  = ((frm_id >> 7) & 0x1);
		frm_id       = (frm_tune << 1) | frm_pol;
		frm->sync         = frm_hdr->sync_word;
		frm->time_tag     = be64toh(frm_hdr->time_tag) - be16toh(frm_hdr->time_offset);
		frm->seq          = frm->time_tag / be16toh(frm_hdr->decimation) / 4096;
		//frm->nsrc         = frm_hdr->nroach;
		frm->nsrc         = _nsrc;
		frm->src          = frm_id - _src0;
		frm->tuning       = be32toh(frm_hdr->tuning_word);
		frm->decimation   = be16toh(frm_hdr->decimation);
		frm->valid_mode   = 1;
		frm->payload_size = pld_size;
		frm->payload_ptr  = frm_pld;
// 		cout << frm_id << frm->src << "valid? " << this->valid_frame(frm) << endl;
		return this->valid_frame(frm);
	}
};

class DRXProcessor {
public:
	inline void operator()(const FrameDesc* frm,
	                       uint64_t         seq0,
	                       uint64_t         nseq_per_obuf,
	                       int              nbuf,
	                       uint8_t*         obufs[],
	                       size_t           ngood_bytes[],
	                       size_t*          src_ngood_bytes[]) {
		int    obuf_idx = ((frm->seq - seq0 >= 1*nseq_per_obuf) +
		(frm->seq - seq0 >= 2*nseq_per_obuf));
		size_t obuf_seq0 = seq0 + obuf_idx*nseq_per_obuf;
		size_t nbyte = frm->payload_size;
		ngood_bytes[obuf_idx]               += nbyte;
		src_ngood_bytes[obuf_idx][frm->src] += nbyte;
		int payload_size = frm->payload_size;
		
		size_t obuf_offset = (frm->seq-obuf_seq0)*frm->nsrc*payload_size;
		
		// Note: Using these SSE types allows the compiler to use SSE instructions
		//         However, they require aligned memory (otherwise segfault)
		uint8_t const* __restrict__ in  = (uint8_t const*)frm->payload_ptr;
		uint8_t*       __restrict__ out = (uint8_t*      )&obufs[obuf_idx][obuf_offset];
		
		int samp = 0;
		for( ; samp<4096; ++samp ) { // HACK TESTING
			out[samp*frm->nsrc + frm->src] = in[samp];
		}
	}
	inline void blank_out_source(uint8_t* data,
	                             int      src,
	                             int      nsrc,
	                             int      nseq) {
		uint8_t* __restrict__ aligned_data = (uint8_t*)data;
		for( int t=0; t<nseq; ++t ) {
			for( int c=0; c<4096; ++c ) {
				aligned_data[t*4096*nsrc + c*nsrc + src] = 0;
				aligned_data[t*4096*nsrc + c*nsrc + src] = 0;
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

class BFdrxreader_impl {
	DRXReaderThread   _reader;
	DRXDecoder         _decoder;
	DRXProcessor       _processor;
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
	int      _slot_nframe;
	BFoffset _seq;
	BFoffset _time_tag;
	int      _chan0;
	int      _chan1;
	int      _payload_size;
	bool     _active;
	uint8_t  _tstate;
	BFdrxreader_sequence_callback _sequence_callback;
	
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
		return _nseq_per_buf * _nsrc * payload_size;
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
				                            _nseq_per_buf);
			}
		}
		_buf_src_ngood_bytes.pop();
		
		_ngood_bytes    += _buf_ngood_bytes.front();
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
		BFoffset    time_tag = _time_tag;
		if( _sequence_callback ) {
			int status = (*_sequence_callback)(seq0,
			                                   _time_tag,
			                                   _chan0,
			                                   _chan1,
			                                   _nsrc,
			                                   &hdr,
			                                   &hdr_size);
			if( status != 0 ) {
				// TODO: What to do here? Needed?
				throw std::runtime_error("BAD HEADER CALLBACK STATUS");
			}
		} else {
			// Simple default for easy testing
			time_tag = _time_tag;
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
	inline BFdrxreader_impl(int    fd,
	                      BFring ring,
	                      int    nsrc,
	                      int    src0,
	                      int    buffer_nframe,
	                      int    slot_nframe,
	                      BFdrxreader_sequence_callback sequence_callback,
	                      int    core)
	:  _reader(fd, nsrc, core, DRX_FRAME_SIZE), _decoder(nsrc, src0), _processor(), 
	   _type_log("lwa_file/type"),
	   _bind_log("lwa_file/bind"),
	   _out_log("lwa_file/out"),
	   _size_log("lwa_file/sizes"),
	   _chan_log("lwa_file/chans"),
	   _stat_log("lwa_file/stats"),
	   _perf_log("lwa_file/perf"), 
	   _nsrc(nsrc), _nseq_per_buf(buffer_nframe), _slot_nframe(slot_nframe),
	   _seq(0), _time_tag(0), _chan0(), _active(false), _tstate(0), 
	   _sequence_callback(sequence_callback),
	   _ring(ring), _oring(_ring),
	   // TODO: Add reset method for stats
	   _ngood_bytes(0), _nmissing_bytes(0) {
		size_t contig_span  = this->bufsize(DRX_FRAME_SIZE - sizeof(drx_hdr_type));
		// Note: 2 write bufs may be open for writing at one time
		size_t total_span   = contig_span * 4;
		size_t nringlet_max = 1;
		_ring.resize(contig_span, total_span, nringlet_max);
		_type_log.update("type : %s", "drx");
		_bind_log.update("ncore : %i\n"
		                 "core0 : %i\n", 
		                 1, core);
		_out_log.update("nring : %i\n"
		                "ring0 : %s\n", 
		                1, _ring.name());
		_size_log.update("nsrc         : %i\n"
		                 "nseq_per_buf : %i\n"
		                 "slot_nframe   : %i\n",
		                 _nsrc, _nseq_per_buf, _slot_nframe);
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
	BFdrxreader_status read() {
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
		
		int state = _reader.run(_seq,
		                        _nseq_per_buf,
		                        _bufs.size(),
		                        buf_ptrs,
		                        ngood_bytes_ptrs,
		                        src_ngood_bytes_ptrs,
		                        &_decoder,
		                        &_processor);
		if( state & DRXReaderThread::READ_ERROR ) {
			return BF_READ_ERROR;
		} else if( state & DRXReaderThread::READ_INTERRUPTED ) {
			return BF_READ_INTERRUPTED;
		}
		const FrameStats* stats = _reader.get_stats();
		_stat_log.update() << "ngood_bytes    : " << _ngood_bytes << "\n"
		                   << "nmissing_bytes : " << _nmissing_bytes << "\n"
		                   << "ninvalid       : " << stats->ninvalid << "\n"
		                   << "ninvalid_bytes : " << stats->ninvalid_bytes << "\n"
		                   << "nlate          : " << stats->nlate << "\n"
		                   << "nlate_bytes    : " << stats->nlate_bytes << "\n"
		                   << "nvalid         : " << stats->nvalid << "\n"
		                   << "nvalid_bytes   : " << stats->nvalid_bytes << "\n";
		
		_t1 = std::chrono::high_resolution_clock::now();
		
		BFdrxreader_status ret;
		bool was_active = _active;
		_active = state & DRXReaderThread::READ_SUCCESS;
		if( _active ) {
			const FrameDesc* frm = _reader.get_last_frame();
			if( frm ) {
				//cout << "Latest nchan, chan0 = " << frm->nchan << ", " << frm->chan0 << endl;
			}
			else {
				//cout << "No latest frame" << endl;
			}
			
			if( frm->src / 2  == 0 ) {
				if( frm->tuning != _chan0 ) {
					_tstate |= 1;
					_chan0 = frm->tuning;
				}
			} else {
				if( frm->tuning != _chan1 ) {
					_tstate |= 2;
					_chan1 = frm->tuning;
				}
			}
			//cout << "State is now " << int(_tstate) << " with " << _chan0 << " and " << _chan1 << " with " << _seq << endl;
			//cout << "  " << frm->time_tag << endl;
			
			if( !was_active ) {
				if( (_tstate == 3 && _nsrc == 4) ||
				    (_tstate != 0 && _nsrc == 2) ) {
					_seq          = round_nearest(frm->seq, _nseq_per_buf);
					_time_tag     = frm->time_tag;
					_payload_size = frm->payload_size;
					_chan_log.update() << "chan0        : " << _chan0 << "\n"
					                   << "chan1        : " << _chan1 << "\n"
					                   << "payload_size : " << _payload_size << "\n";
					this->begin_sequence();
					ret = BF_READ_STARTED;
					_tstate = 0;
				} else {
					ret = BF_READ_NO_DATA;
					_active = false;
					_reader.reset_last_frame();
				}
			} else {
				//cout << "Continuing data, seq = " << seq << endl;
				if( (_tstate == 3 && _nsrc == 4) ||
				    (_tstate != 0 && _nsrc == 2) ) {
					_time_tag     = frm->time_tag;
					_payload_size = frm->payload_size;
					_chan_log.update() << "chan0        : " << _chan0 << "\n"
					                   << "chan1        : " << _chan1 << "\n"
					                   << "payload_size : " << _payload_size << "\n";
					this->end_sequence();
					this->begin_sequence();
					ret = BF_READ_CHANGED;
					_tstate = 0;
				} else {
					ret = BF_READ_CONTINUED;
				}
			}
			
			if( ret != BF_READ_NO_DATA ) {
				if( _bufs.size() == 2 ) {
					this->commit_buf();
				}
				this->reserve_buf();
			}
		} else {
			
			if( was_active ) {
				this->flush();
				ret = BF_READ_ENDED;
			} else {
				ret = BF_READ_NO_DATA;
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

BFstatus bfDrxReaderCreate(BFdrxreader* obj,
                           int           fd,
                           BFring        ring,
                           BFsize        nsrc,
                           BFsize        src0,
                           BFsize        buffer_nframe,
                           BFsize        slot_nframe,
                           BFdrxreader_sequence_callback sequence_callback,
                           int           core) {
	BF_ASSERT(obj, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*obj = new BFdrxreader_impl(fd, ring, nsrc, src0,
		                                          buffer_nframe, slot_nframe,
		                                          sequence_callback, core),
		              *obj = 0);
}
BFstatus bfDrxReaderDestroy(BFdrxreader obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	delete obj;
	return BF_STATUS_SUCCESS;
}
BFstatus bfDrxReaderRead(BFdrxreader obj, BFdrxreader_status* result) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*result = obj->read(),
	                   *result = BF_READ_ERROR);
}
BFstatus bfDrxReaderFlush(BFdrxreader obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(obj->flush());
}
BFstatus bfDrxReaderEnd(BFdrxreader obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(obj->end_writing());
}