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
 
#include "data_capture.hpp"

#if BF_HWLOC_ENABLED
int HardwareLocality::bind_memory_to_core(int core) {
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
#endif // BF_HWLOC_ENABLED

BFdatacapture_status BFdatacapture_impl::recv() {
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
	if( state & DataCaptureThread::CAPTURE_ERROR ) {
		return BF_CAPTURE_ERROR;
	} else if( state & DataCaptureThread::CAPTURE_INTERRUPTED ) {
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
	
	BFoffset seq0, time_tag;
	const void* hdr;
	size_t hdr_size;
	
	BFdatacapture_status ret;
	bool was_active = _active;
	_active = state & DataCaptureThread::CAPTURE_SUCCESS;
	if( _active ) {
	    const PacketDesc* pkt = _capture.get_last_packet();
	    this->on_sequence_active(pkt);
		if( !was_active ) {
			//cout << "Beginning of sequence, first pkt seq = " << pkt->seq << endl;
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

BFstatus bfDataCaptureDestroy(BFdatacapture obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	delete obj;
	return BF_STATUS_SUCCESS;
}

BFstatus bfDataCaptureRecv(BFdatacapture obj, BFdatacapture_status* result) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN_ELSE(*result = obj->recv(),
	                   *result = BF_CAPTURE_ERROR);
}

BFstatus bfDataCaptureFlush(BFdatacapture obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(obj->flush());
}

BFstatus bfDataCaptureEnd(BFdatacapture obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(obj->end_writing());
}

BFstatus bfDataCaptureCallbackCreate(BFdatacapture_callback* obj) {
	BF_TRY_RETURN_ELSE(*obj = new BFdatacapture_callback_impl(),
                               *obj = 0);
}

BFstatus bfDataCaptureCallbackSetCHIPS(BFdatacapture_callback obj, BFdatacapture_chips_sequence_callback callback) {
	obj->set_chips(callback);
	return BF_STATUS_SUCCESS;
}

BFstatus bfDataCaptureCallbackSetTBN(BFdatacapture_callback obj, BFdatacapture_tbn_sequence_callback callback) {
	obj->set_tbn(callback);
	return BF_STATUS_SUCCESS;
}

BFstatus bfDataCaptureCallbackSetDRX(BFdatacapture_callback obj, BFdatacapture_drx_sequence_callback callback) {
	obj->set_drx(callback);
	return BF_STATUS_SUCCESS;
}

