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
#include <bifrost/address.h>

#include <netinet/in.h>

BFstatus bfAddressCreate(BFaddress*  addr,
                         const char* addr_string,
                         int         port,
                         unsigned    family) {
	BF_ASSERT(addr, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*addr = (BFaddress) new sockaddr_storage(Socket::address(addr_string,
	                                                            port,
	                                                            family)),
	                   *addr = 0);
}
BFstatus bfAddressDestroy(BFaddress addr) {
	BF_ASSERT(addr, BF_STATUS_INVALID_HANDLE);
	delete addr;
	return BF_STATUS_SUCCESS;
}
BFstatus bfAddressGetFamily(BFaddress addr, unsigned* family) {
	BF_ASSERT(addr, BF_STATUS_INVALID_HANDLE);
	*family = addr->sa_family;
	return BF_STATUS_SUCCESS;
}
BFstatus bfAddressGetPort(BFaddress addr, int* port) {
	BF_ASSERT(addr, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(port, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(addr->sa_family == AF_INET ||
	          addr->sa_family == AF_INET6,
	          BF_STATUS_INVALID_ARGUMENT);
	BF_TRY_RETURN_ELSE(*port = ntohs(((sockaddr_in*)addr)->sin_port),
	                   *port = 0);
}
BFstatus bfAddressGetMTU(BFaddress addr, int* mtu) {
	BF_ASSERT(addr, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(mtu,  BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*mtu = Socket::discover_mtu(*(sockaddr_storage*)addr),
	                   *mtu = 0);
}
BFstatus bfAddressGetString(BFaddress addr,
                            BFsize    bufsize, // 128 should always be enough
                            char*     buf) {
	BF_ASSERT(addr, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(buf,  BF_STATUS_INVALID_POINTER);
	if( !bufsize ) {
		return BF_STATUS_SUCCESS;
	}
	buf[bufsize-1] = 0; // Make sure buffer is always NULL-terminated
	BF_TRY_RETURN_ELSE(std::strncpy(buf, Socket::address_string(*(sockaddr_storage*)addr).c_str(), bufsize-1),
	                   *buf = 0);
}
