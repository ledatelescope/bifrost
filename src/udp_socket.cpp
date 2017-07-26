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
#include "Socket.hpp"
#include <bifrost/udp_socket.h>
#include <bifrost/address.h>

struct BFudpsocket_impl : public Socket {
	BFudpsocket_impl() : Socket(SOCK_DGRAM) {}
};

BFstatus bfUdpSocketCreate(BFudpsocket* obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*obj = (BFudpsocket)new BFudpsocket_impl(),//BFudpsocket_impl::SOCK_DGRAM),
	                   *obj = 0);
}
BFstatus bfUdpSocketDestroy(BFudpsocket obj) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	delete obj;
	return BF_STATUS_SUCCESS;
}
BFstatus bfUdpSocketConnect(BFudpsocket obj, BFaddress remote_addr) {
	BF_ASSERT(obj,         BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(remote_addr, BF_STATUS_INVALID_ARGUMENT);
	BF_TRY_RETURN(obj->connect(*(sockaddr_storage*)remote_addr));
}
BFstatus bfUdpSocketBind(   BFudpsocket obj, BFaddress local_addr) {
	BF_ASSERT(obj,        BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(local_addr, BF_STATUS_INVALID_ARGUMENT);
	BF_TRY_RETURN(obj->bind(*(sockaddr_storage*)local_addr));
}
BFstatus bfUdpSocketShutdown(BFudpsocket obj) {
	BF_ASSERT(obj,        BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(obj->shutdown());
}
BFstatus bfUdpSocketClose(BFudpsocket obj) {
	BF_ASSERT(obj,        BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(obj->close());
}
BFstatus bfUdpSocketSetTimeout(BFudpsocket obj, double secs) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_TRY_RETURN(obj->set_timeout(secs));
}
BFstatus bfUdpSocketGetTimeout(BFudpsocket obj, double* secs) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	// WAR for old Socket implem returning Socket::Error not BFexception
	try {
		*secs = obj->get_timeout();
	}
	catch( Socket::Error ) {
		*secs = 0;
		return BF_STATUS_INVALID_STATE;
	}
	catch(...) {
		*secs = 0;
		return BF_STATUS_INTERNAL_ERROR;
	}
	return BF_STATUS_SUCCESS;
	//BF_TRY(*secs = obj->get_timeout(),
	//       *secs = 0);
}
BFstatus bfUdpSocketGetMTU(BFudpsocket obj, int* mtu) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(mtu, BF_STATUS_INVALID_POINTER);
	// WAR for old Socket implem returning Socket::Error not BFexception
	try {
		*mtu = obj->get_mtu();
	}
	catch( Socket::Error ) {
		*mtu = 0;
		return BF_STATUS_INVALID_STATE;
	}
	catch(...) {
		*mtu = 0;
		return BF_STATUS_INTERNAL_ERROR;
	}
	return BF_STATUS_SUCCESS;
	//BF_TRY(*mtu = obj->get_mtu(),
	//       *mtu = 0);
}
BFstatus bfUdpSocketGetFD(BFudpsocket obj, int* fd) {
	BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(fd,  BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*fd = obj->get_fd(),
	                   *fd = 0);
}
