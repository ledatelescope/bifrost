/*
 * Copyright (c) 2019-2021, The Bifrost Authors. All rights reserved.
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
 
#include "packet_capture.hpp"

#define BF_JAYCE_DEBUG 0

#if BF_JAYCE_DEBUG
#define BF_PRINTD(stmt) \
    std::cout << stmt << std::endl
#else // not BF_JAYCE_DEBUG
#define BF_PRINTD(stmt)
#endif

// Reads, decodes and unpacks frames into the provided buffers
// Note: Read continues until the first frame that belongs
//         beyond the end of the provided buffers. This frame is
//         saved, accessible via get_last_frame(), and will be
//         processed on the next call to run() if possible.
template<class PDC, class PPC>
int PacketCaptureThread::run(uint64_t seq_beg,
	                         uint64_t nseq_per_obuf,
	                         int      nbuf,
	                         uint8_t* obufs[],
	                         size_t*  ngood_bytes[],
	                         size_t*  src_ngood_bytes[],
	                         PDC*     decode,
	                         PPC*     process) {
    uint64_t seq_end = seq_beg + nbuf*nseq_per_obuf;
	size_t local_ngood_bytes[2] = {0, 0};
	int ret;
	while( true ) {
		if( !_have_pkt ) {
			uint8_t* pkt_ptr;
			int pkt_size = _method->recv_packet(&pkt_ptr);
			if( pkt_size <= 0 ) {
				if( errno == EAGAIN || errno == EWOULDBLOCK ) {
					ret = CAPTURE_TIMEOUT; // Timed out
				} else if( errno == EINTR ) {
					ret = CAPTURE_INTERRUPTED; // Interrupted by signal
				} else if( pkt_size == 0 ) {
				    ret = CAPTURE_NO_DATA;
				} else {
					ret = CAPTURE_ERROR; // Socket error
				}
				break;
			}
			BF_PRINTD("HERE");
			if( !(*decode)(pkt_ptr, pkt_size, &_pkt) ) {
			    BF_PRINTD("INVALID " << std::hex << _pkt.sync << " " << std::dec << _pkt.src << " " << _pkt.src << " " << _pkt.time_tag << " " << _pkt.tuning << " " << _pkt.valid_mode);
	            ++_stats.ninvalid;
				_stats.ninvalid_bytes += pkt_size;
				continue;
			}
			BF_PRINTD("VALID " << std::hex << _pkt.sync << " " << std::dec << _pkt.src << " " << _pkt.src << " " << _pkt.time_tag << " " << _pkt.tuning << " " << _pkt.valid_mode);
			_have_pkt = true;
		}
		BF_PRINTD("NOW" << " " << _pkt.seq << " >= " << seq_end);
		if( greater_equal(_pkt.seq, seq_end) ) {
			// Reached the end of this processing gulp, so leave this
			//   packet unprocessed and return.
			ret = CAPTURE_SUCCESS;
			BF_PRINTD("BREAK NOW");
			break;
		}
		BF_PRINTD("HERE" << " " << _pkt.seq << " < " << seq_beg);
		_have_pkt = false;
		if( less_than(_pkt.seq, seq_beg) ) {
			++_stats.nlate;
			_stats.nlate_bytes += _pkt.payload_size;
			++_src_stats[_pkt.src].nlate;
			_src_stats[_pkt.src].nlate_bytes += _pkt.payload_size;
			BF_PRINTD("CONTINUE HERE");
			continue;
		}
		BF_PRINTD("FINALLY");
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

BFstatus bfPacketCaptureCallbackCreate(BFpacketcapture_callback* obj) {
    BF_TRY_RETURN_ELSE(*obj = new BFpacketcapture_callback_impl(),
                       *obj = 0);
}

BFstatus bfPacketCaptureCallbackDestroy(BFpacketcapture_callback obj) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    delete obj;
    return BF_STATUS_SUCCESS;
}

BFstatus bfPacketCaptureCallbackSetCHIPS(BFpacketcapture_callback obj,
                                         BFpacketcapture_chips_sequence_callback callback) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_chips(callback);
    return BF_STATUS_SUCCESS;
}

BFstatus bfPacketCaptureCallbackSetIBeam(BFpacketcapture_callback obj,
                                         BFpacketcapture_ibeam_sequence_callback callback) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_ibeam(callback);
    return BF_STATUS_SUCCESS;
}

BFstatus bfPacketCaptureCallbackSetPBeam(BFpacketcapture_callback obj,
                                         BFpacketcapture_pbeam_sequence_callback callback) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_pbeam(callback);
    return BF_STATUS_SUCCESS;
}

BFstatus bfPacketCaptureCallbackSetCOR(BFpacketcapture_callback obj,
                                       BFpacketcapture_cor_sequence_callback callback) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_cor(callback);
    return BF_STATUS_SUCCESS;
}

BFstatus bfPacketCaptureCallbackSetVDIF(BFpacketcapture_callback obj,
                                        BFpacketcapture_vdif_sequence_callback callback) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_vdif(callback);
    return BF_STATUS_SUCCESS;
}

BFstatus bfPacketCaptureCallbackSetTBN(BFpacketcapture_callback obj,
                                       BFpacketcapture_tbn_sequence_callback callback) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_tbn(callback);
    return BF_STATUS_SUCCESS;
}

BFstatus bfPacketCaptureCallbackSetDRX(BFpacketcapture_callback obj,
                                       BFpacketcapture_drx_sequence_callback callback) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    obj->set_drx(callback);
    return BF_STATUS_SUCCESS;
}

BFpacketcapture_status BFpacketcapture_impl::recv() {
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
	
	int state = _capture->run(_seq,
	                          _nseq_per_buf,
	                          _bufs.size(),
	                          buf_ptrs,
	                          ngood_bytes_ptrs,
	                          src_ngood_bytes_ptrs,
	                          _decoder,
	                          _processor);
	BF_PRINTD("OUTSIDE");
	if( state & PacketCaptureThread::CAPTURE_ERROR ) {
	    return BF_CAPTURE_ERROR;
	} else if( state & PacketCaptureThread::CAPTURE_INTERRUPTED ) {
		return BF_CAPTURE_INTERRUPTED;
	} else if( state & PacketCaptureThread::CAPTURE_INTERRUPTED ) {
	    if( _active ) {
    	    return BF_CAPTURE_ENDED;
    	} else {
    	    return BF_CAPTURE_NO_DATA;
    	}
	}
	const PacketStats* stats = _capture->get_stats();
	_stat_log.update() << "ngood_bytes    : " << _ngood_bytes << "\n"
	                   << "nmissing_bytes : " << _nmissing_bytes << "\n"
	                   << "ninvalid       : " << stats->ninvalid << "\n"
	                   << "ninvalid_bytes : " << stats->ninvalid_bytes << "\n"
	                   << "nlate          : " << stats->nlate << "\n"
	                   << "nlate_bytes    : " << stats->nlate_bytes << "\n"
	                   << "nvalid         : " << stats->nvalid << "\n"
	                   << "nvalid_bytes   : " << stats->nvalid_bytes << "\n";
	
	_t1 = std::chrono::high_resolution_clock::now();
	
	BFoffset seq0, time_tag=0;
	const void* hdr=NULL;
	size_t hdr_size=0;
	
	BFpacketcapture_status ret;
	bool was_active = _active;
	_active = state & PacketCaptureThread::CAPTURE_SUCCESS;
	BF_PRINTD("ACTIVE: " << _active << " WAS ACTIVE: " << was_active);
	if( _active ) {
	    BF_PRINTD("START");
	    const PacketDesc* pkt = _capture->get_last_packet();
	    BF_PRINTD("PRE-CALL");
	    this->on_sequence_active(pkt);
	    BF_PRINTD("POST-CALL");
		if( !was_active ) {
			BF_PRINTD("Beginning of sequence, first pkt seq = " << pkt->seq);
			this->on_sequence_start(pkt, &seq0, &time_tag, &hdr, &hdr_size);
			this->begin_sequence(seq0, time_tag, hdr, hdr_size);
			ret = BF_CAPTURE_STARTED;
		} else {
			//cout << "Continuing data, seq = " << seq << endl;
			if( this->has_sequence_changed(pkt) ) {
			    this->on_sequence_changed(pkt, &seq0, &time_tag, &hdr, &hdr_size);
			    this->end_sequence();
				this->begin_sequence(seq0, time_tag, hdr, hdr_size);
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

BFstatus bfDiskReaderCreate(BFpacketcapture* obj,
                            const char*      format,
                            int              fd,
                            BFring           ring,
                            BFsize           nsrc,
                            BFsize           src0,
                            BFsize           buffer_ntime,
                            BFsize           slot_ntime,
                            BFpacketcapture_callback sequence_callback,
                            int              core) {
    return BFpacketcapture_create(obj,
                                  format,
                                  fd,
                                  ring,
                                  nsrc,
                                  src0,
                                  buffer_ntime,
                                  slot_ntime,
                                  sequence_callback,
                                  core,
                                  BF_IO_DISK);
}

BFstatus bfUdpCaptureCreate(BFpacketcapture* obj,
                            const char*      format,
                            int              fd,
                            BFring           ring,
                            BFsize           nsrc,
                            BFsize           src0,
                            BFsize           max_payload_size,
                            BFsize           buffer_ntime,
                            BFsize           slot_ntime,
                            BFpacketcapture_callback sequence_callback,
                            int              core) {
    return BFpacketcapture_create(obj,
                                  format,
                                  fd,
                                  ring,
                                  nsrc,
                                  src0,
                                  buffer_ntime,
                                  slot_ntime,
                                  sequence_callback,
                                  core,
                                  BF_IO_UDP);
}

BFstatus bfUdpSnifferCreate(BFpacketcapture* obj,
                            const char*      format,
                            int              fd,
                            BFring           ring,
                            BFsize           nsrc,
                            BFsize           src0,
                            BFsize           max_payload_size,
                            BFsize           buffer_ntime,
                            BFsize           slot_ntime,
                            BFpacketcapture_callback sequence_callback,
                            int              core) {
    return BFpacketcapture_create(obj,
                                  format,
                                  fd,
                                  ring,
                                  nsrc,
                                  src0,
                                  buffer_ntime,
                                  slot_ntime,
                                  sequence_callback,
                                  core,
                                  BF_IO_SNIFFER);
}

BFstatus bfPacketCaptureDestroy(BFpacketcapture obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	delete obj;
	return BF_STATUS_SUCCESS;
}

BFstatus bfPacketCaptureRecv(BFpacketcapture obj,
                             BFpacketcapture_status* result) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*result = obj->recv(),
	                   *result = BF_CAPTURE_ERROR);
}

BFstatus bfPacketCaptureFlush(BFpacketcapture obj) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    BF_TRY_RETURN(obj->flush());
}

BFstatus bfPacketCaptureSeek(BFpacketcapture obj, BFoffset offset, BFiowhence whence, BFoffset* position) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(*position = obj->seek(offset, whence));
}

BFstatus bfPacketCaptureTell(BFpacketcapture obj, BFoffset* position) {
    BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
    BF_TRY_RETURN(*position = obj->tell());
}

BFstatus bfPacketCaptureEnd(BFpacketcapture obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(obj->end_writing());
}
