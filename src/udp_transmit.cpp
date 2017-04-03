/*
 * Copyright (c) 2017, The Bifrost Authors. All rights reserved.
 * Copyright (c) 2017, The University of New Mexico. All rights reserved.
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
#include <bifrost/udp_transmit.h>
#include <bifrost/affinity.h>
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

struct PacketStats {
	size_t ninvalid;
	size_t ninvalid_bytes;
	size_t nlate;
	size_t nlate_bytes;
	size_t nvalid;
	size_t nvalid_bytes;
};

class UDPTransmitThread : public BoundThread {
	PacketStats       _stats;
	
	int               _fd;
	
public:
	UDPTransmitThread(int fd, int core=0)
		: BoundThread(core), _fd(fd) {
		this->reset_stats();
	}
	inline ssize_t send(msghdr* packet) {
		ssize_t nsent = sendmsg(_fd, packet, 0);
		if( nsent == -1 ) {
			++_stats.ninvalid;
			_stats.ninvalid_bytes += packet->msg_iovlen;
		} else {
			++_stats.nvalid;
			_stats.nvalid_bytes += nsent;
		}
		return nsent;
	}
	inline ssize_t sendmany(mmsghdr *packets, unsigned int npackets) {
		ssize_t nsent = sendmmsg(_fd, packets, npackets, 0);
		if( nsent == -1 ) {
			_stats.ninvalid += npackets;
			_stats.ninvalid_bytes += npackets * packets->msg_len;
		} else {
			_stats.nvalid += npackets;
			_stats.nvalid_bytes += npackets * packets->msg_len;
		}
		return nsent;
	}
	inline const PacketStats* get_stats() const { return &_stats; }
	inline void reset_stats() {
		::memset(&_stats, 0, sizeof(_stats));
	}
};

class BFudptransmit_impl {
	UDPTransmitThread  _transmit;
	ProcLog            _type_log;
	ProcLog            _bind_log;
	ProcLog            _stat_log;
	pid_t              _pid;
	
	void update_stats_log() {
		const PacketStats* stats = _transmit.get_stats();
		_stat_log.update() << "ngood_bytes    : " << stats->nvalid_bytes << "\n"
		                   << "nmissing_bytes : " << stats->ninvalid_bytes << "\n"
		                   << "ninvalid       : " << stats->ninvalid << "\n"
		                   << "ninvalid_bytes : " << stats->ninvalid_bytes << "\n"
		                   << "nlate          : " << stats->nlate << "\n"
		                   << "nlate_bytes    : " << stats->nlate_bytes << "\n"
		                   << "nvalid         : " << stats->nvalid << "\n"
		                   << "nvalid_bytes   : " << stats->nvalid_bytes << "\n";
	}
public:
	inline BFudptransmit_impl(int fd,
	                          int core)
		: _transmit(fd, core),
		  _type_log("udp_transmit/type"),
		  _bind_log("udp_transmit/bind"),
		  _stat_log("udp_transmit/stats") {
		_type_log.update() << "type : " << "generic";
		_bind_log.update() << "ncore : " << 1 << "\n"
		                   << "core0 : " << core << "\n";
	}
	BFudptransmit_status send(char *packet, unsigned int len) {
		ssize_t state;
		struct msghdr msg;
		struct iovec iov[1];
		
		memset(&msg, 0, sizeof(msg));
		msg.msg_iov = iov;
		msg.msg_iovlen = 1;
		iov[0].iov_base = packet;
		iov[0].iov_len = len;
		
		state = _transmit.send( &msg );
		if( state == -1 ) {
			return BF_TRANSMIT_ERROR;
		}
		this->update_stats_log();
		return BF_TRANSMIT_CONTINUED;
	}
	BFudptransmit_status sendmany(char *packets, unsigned int len, unsigned int npackets) {
		ssize_t state;
		unsigned int i;
		struct mmsghdr *mmsg = NULL;
		struct iovec *iovs = NULL;
		
		mmsg = (struct mmsghdr *) malloc(sizeof(struct mmsghdr)*npackets);
		iovs = (struct iovec *) malloc(sizeof(struct iovec)*npackets);
		memset(mmsg, 0, sizeof(struct mmsghdr)*npackets);
		for(i=0; i<npackets; i++) {
			mmsg[i].msg_hdr.msg_iov = &iovs[i];
			mmsg[i].msg_hdr.msg_iovlen = 1;
			iovs[i].iov_base = (packets + i*len);
			iovs[i].iov_len = len;
		}
		
		state = _transmit.sendmany(mmsg, npackets);
		free(mmsg);
		free(iovs);
		if( state == -1 ) {
			return BF_TRANSMIT_ERROR;
		}
		this->update_stats_log();
		return BF_TRANSMIT_CONTINUED;
	}
};

BFstatus bfUdpTransmitCreate(BFudptransmit* obj,
                            int           fd,
                            int           core) {
	BF_ASSERT(obj, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*obj = new BFudptransmit_impl(fd, core),
		              *obj = 0);

}
BFstatus bfUdpTransmitDestroy(BFudptransmit obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	delete obj;
	return BF_STATUS_SUCCESS;
}
BFstatus bfUdpTransmitSend(BFudptransmit obj, char* packet, unsigned int len) {
	BF_TRY_RETURN(obj->send(packet, len));
}
BFstatus bfUdpTransmitSendMany(BFudptransmit obj, char* packets, unsigned int len, unsigned int npackets) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(obj->sendmany(packets, len, npackets));
}
