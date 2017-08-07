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

/*
  Convenient socket wrapper providing efficient bulk packet tx/rx
  
  Features: Supports UDP (SOCK_DGRAM) and TCP (SOCK_STREAM) socket types
            Supports IPv4, IPv6 and UNIX socket families
            Convenient address specification by string path, IP, interface or host name
            Sets large socket buffer sizes by default
            Send or receive single packets or blocks of many packets with one call
            Timeouts on send/recv calls
            Dropped packet counter
            MTU discovery
            Errors returned as exceptions
            
  Performance tuning tips:
    Bind socket thread to a single CPU core to achieve peak performance
    Set the SO_REUSEPORT option to allow multithreaded capture
      See https://lwn.net/Articles/542629/
    Increase the max SO_SNDBUF/SO_RCVBUF limits via the files:
      /proc/sys/net/core/wmem_max (for SO_SNDBUF)
      /proc/sys/net/core/rmem_max (for SO_RCVBUF)
      Note that the kernel actually allocates double the amount requested
        with setsockopt().
      An alternative method is to give the process CAP_NET_ADMIN privilages
        and use the SO_RCVBUFFORCE/SO_SNDBUFFORCE options.
        $ sudo setcap cap_net_admin=eip myexecutable
        
  Useful socket options (see http://man7.org/linux/man-pages/man7/socket.7.html):
    SO_BROADCAST:    Enable sending UDP packets to broadcast addresses
    SO_BINDTODEVICE: Force packets to be sent/received via a specific eth device
    
 TODO: Need to support non-blocking connect() etc.?
       Think about whether need to add support for recvmsg/recvfrom/sendmsg/sendto etc.
       [DONE]Work out how to handle detection of remotely-closed TCP connections
       [DONE]Work out how to use REUSEADDR and SO_LINGER
       [DONE]Work out how to handle accept() calls
       [DONE]Study changes required to support IPv6
       [DONE]Study changes required to support UNIX sockets

 TODO: Most efficient way to implement all-to-all will be to use a single
         bind() socket for each process and pass destination addresses to send_block().
         This will mean that only a single set of large tx/rx buffers are allocated
           per process.
           Using TCP, where there is one socket per process-pair, is likely to
             be very innefficient and to scale very badly.
alltoall: Split local buffer up into Nrecv pieces, each with a different dest addr
            Then split each piece up into packets, each with the seq no. and local proc's id
         
  A note on address structures:
    IPv4 (AF_INET):  sockaddr_in
    IPv6 (AF_INET6): sockaddr_in6
    UNIX (AF_UNIX):  sockaddr_un
    Any pointer: *sockaddr
    Any storage:  sockaddr_storage

// UDP server
// ----------
Socker server(SOCK_DGRAM);
server.listen(Socket::address(local_addr, port));
server.send/recv_block/packet(...);

// TCP server
// ----------
Socket server(SOCK_STREAM);
server.listen(Socket::address(local_addr, port));
//auto   client = std::shared_ptr<Socket>(server.accept());  // C++11
//Socket client = *std::shared_ptr<Socket>(server.accept()); // C++11
//Socket client = server.accept(); // C++11 (with minor modifications to API)
Socket* client = server.accept();
client->send/recv_block/packet(...);
delete client;

// UDP/TCP client
// --------------
Socket client(SOCK_DGRAM/SOCK_STREAM);
client.connect(Socket::address(remote_addr, port));
client.connect(Socket::any_address()); // UDP only
client.send/recv_block/packet(...);
*/

#pragma once

#include <stdexcept>
#include <string>
#include <stdint.h>
#include <cstring>
#include <climits>
#include <vector>
#include <sstream>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>
#include <sys/un.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <ifaddrs.h>
//#include <linux/net.h>

class Socket {
	// Not copy-assignable
	Socket(Socket const& );
	Socket& operator=(Socket const& );
#if __cplusplus >= 201103L
	inline void replace(Socket& s);
#endif
	// Manage an existing socket descriptor
	// Note: Accessible only via the named constructor Socket::manage
	struct ManageTag {};
	inline Socket(int fd, ManageTag );
public:
	// TODO: Move this definition below
	class Error : public std::runtime_error {
		typedef std::runtime_error super_t;
	protected:
		virtual const char* what() const throw() {
			return super_t::what();
		}
	public:
		Error(const std::string& what_arg)
			: super_t(what_arg) {}
	};
	enum {
		DEFAULT_SOCK_BUF_SIZE  = 256*1024*1024,
		DEFAULT_LINGER_SECS    = 3,
		DEFAULT_MAX_CONN_QUEUE = 128
	};
	enum sock_type {
		SOCK_DGRAM  = ::SOCK_DGRAM,
		SOCK_STREAM = ::SOCK_STREAM
	};
	// Manage an existing socket (usually one returned by Socket::accept())
	// TODO: With C++11 this could return by value (moved), which would be nicer
	inline static Socket* manage(int fd) { return new Socket(fd, ManageTag()); }
	inline explicit       Socket(/*sock_type*/int type=SOCK_DGRAM)
		: _fd(-1), _type((sock_type)type), _family(AF_UNSPEC),
		  _mode(Socket::MODE_CLOSED) {}
	
	virtual ~Socket() { this->close(); }
#if __cplusplus >= 201103L
	// Move semantics
	inline Socket(Socket&& s)                           { this->replace(s); }
	inline Socket& operator=(Socket&& s) { this->close(); this->replace(s); return *this; }
#endif
	inline void swap(Socket& s);
	// Address generator
	// Note: Supports UNIX paths, IPv4 and IPv6 addrs, interfaces and hostnames
	//       Passing addr=0  means "any address"
	//       Passing port=-1 implies family=AF_UNIX
	inline static sockaddr_storage address(std::string    addr,
	                                // Note: int so that -1 can be given
	                                int            port,
	                                sa_family_t    family=AF_UNSPEC);
	inline static sockaddr_storage any_address(sa_family_t family=AF_UNSPEC);
	inline static std::string      address_string(sockaddr_storage addr);
	inline static int              discover_mtu(sockaddr_storage remote_address);
	
	// Server initialisation
	inline void bind(sockaddr_storage local_address,
	          int              max_conn_queue=DEFAULT_MAX_CONN_QUEUE);
	// Client initialisation
	inline void connect(sockaddr_storage remote_address);
	// Accept incoming SOCK_STREAM connection requests
	// TODO: With C++11 this could return by value (moved), which would be nicer
	inline Socket* accept(double timeout_secs=-1);
	// Note: This can be used to unblock recv calls from another thread
	//         This behaviour is not explicitly documented, but it works, is
	//           much simpler than having to mess around with select/poll, and
	//           is better than relying on timeouts.
	//           IMHO this should be official behaviour of POSIX shutdown!
	inline void shutdown(int how=SHUT_RD);
	//void shutdown(int how=SHUT_RDWR);
	inline void close();
	// Send/receive
	// Note: These four methods return the number of packets received/sent
	inline size_t recv_block(size_t            npacket,       // Max for UDP
	                  void*             header_buf,    // Can be NULL
	                  size_t const*     header_offsets,
	                  size_t const*     header_sizes,
	                  void*             payload_buf,
	                  size_t const*     payload_offsets,
	                  size_t const*     payload_sizes, // Max for UDP
	                  size_t*           packet_sizes,
	                  sockaddr_storage* packet_sources=0,
	                  double            timeout_secs=-1);
	inline size_t recv_packet(void*             header_buf,
	                   size_t            header_size,
	                   void*             payload_buf,
	                   size_t            payload_size,
	                   size_t*           packet_size,
	                   sockaddr_storage* packet_source=0,
	                   double            timeout_secs=-1);
	// No. dropped packets detected during last call to recv_*
	inline size_t get_drop_count() const { return _ndropped; }
	// No. bytes received by last call to recv_*
	// Note: Only valid if packet_sizes was non-NULL, otherwise returns 0
	inline size_t get_recv_size()  const { return _nrecv_bytes; }
	inline size_t send_block(size_t                  npacket,
	                  void   const*           header_buf,
	                  size_t const*           header_offsets,
	                  size_t const*           header_sizes,
	                  void   const*           payload_buf,
	                  size_t const*           payload_offsets,
	                  size_t const*           payload_sizes,
	                  sockaddr_storage const* packet_dests=0, // Not needed after connect()
	                  double                  timeout_secs=-1);
	inline size_t send_packet(void const*             header_buf,
	                   size_t                  header_size,
	                   void const*             payload_buf,
	                   size_t                  payload_size,
	                   sockaddr_storage const* packet_dest=0, // Not needed after connect()
	                   double                  timeout_secs=-1);
	inline sockaddr_storage get_local_address()  /*const*/; // check_error is non-const
	inline sockaddr_storage get_remote_address() /*const*/;
	inline int       get_mtu() /*const*/ {
		if( _mode != Socket::MODE_CONNECTED ) {
			throw Socket::Error("Not connected");
		}
		return this->get_option<int>(IP_MTU, IPPROTO_IP);
	}
	template<typename T>
	inline void set_option(int optname, T value, int level=SOL_SOCKET) {
		//::setsockopt(_fd, level, optname, &value, sizeof(value));
		check_error( ::setsockopt(_fd, level, optname, &value, sizeof(value)),
		             "set socket option" );
	}
	// Note: non-const because check_error closes the socket on failure
	template<typename T>
	inline T get_option(int optname, int level=SOL_SOCKET) /*const*/ {
		T value;
		socklen_t size = sizeof(value);
		check_error( ::getsockopt(_fd, level, optname, &value, &size),
		             "get socket option");
		return value;
	}
	inline int get_fd() const { return _fd; }
	inline void set_timeout(double secs) {
		if( secs > 0 ) {
			timeval timeout;
			timeout.tv_sec  = (int)secs;
			timeout.tv_usec = (int)((secs - timeout.tv_sec)*1e6);
			this->set_option(SO_RCVTIMEO, timeout);
			this->set_option(SO_SNDTIMEO, timeout);
		}
	}
	inline double get_timeout() const {
		// WAR for non-const get_option (which is because of close-on-error)
		Socket* self = const_cast<Socket*>(this);
		// TODO: This ignores SO_SNDTIMEO
		timeval timeout = self->get_option<timeval>(SO_RCVTIMEO);
		return timeout.tv_sec + timeout.tv_usec*1e-6;
	}
private:
	inline void open(sa_family_t family);
	inline void set_default_options();
	inline void check_error(int retval, std::string what) {
		if( retval < 0 ) {
			if( errno == ENOTCONN ) {
				this->close();
			}
			std::stringstream ss;
			ss << "Failed to " << what << ": (" << errno << ") "
			   << strerror(errno);
			throw Socket::Error(ss.str());
		}
	}
	inline void prepare_msgs(size_t            npacket,
	                  void*             header_buf,
	                  size_t const*     header_offsets,
	                  size_t const*     header_sizes,
	                  void*             payload_buf,
	                  size_t const*     payload_offsets,
	                  size_t const*     payload_sizes,
	                  sockaddr_storage* packet_addrs);
	inline static int addr_from_hostname(const char* hostname,
	                              sockaddr*   address,
	                              sa_family_t family=AF_UNSPEC,
	                              int         socktype=0);
	inline static int addr_from_interface(const char* ifname,
	                               sockaddr*   address,
	                               sa_family_t family=AF_UNSPEC);
	int         _fd;
	sock_type   _type;
	sa_family_t _family;
	enum {
		MODE_CLOSED,
		MODE_BOUND,
		MODE_LISTENING,
		MODE_CONNECTED
	} _mode;
	// Logged quantities
	size_t _ndropped;
	size_t _nrecv_bytes;
	// Temporary arrays
	std::vector<mmsghdr> _msgs;
	std::vector<iovec>   _iovecs;
};

sockaddr_storage Socket::any_address(sa_family_t family) {
	//return Socket::address(0, -1, family);
	return Socket::address("", 0, family);
}
sockaddr_storage Socket::address(//const char*    addrstr,
                                 std::string    addrstr,
                                 int            port,
                                 sa_family_t    family) {
	sockaddr_storage sas;
	memset(&sas, 0, sizeof(sas));
	sockaddr_in*  sa4 = reinterpret_cast<sockaddr_in*> (&sas);
	sockaddr_in6* sa6 = reinterpret_cast<sockaddr_in6*>(&sas);
	sockaddr_un*  saU = reinterpret_cast<sockaddr_un*> (&sas);
	//if( !addrstr || !std::strlen(addrstr) ) {
	if( addrstr.empty() ) {
		// No address means "any address"
		sas.ss_family = family;
		switch( family ) {
		case AF_INET: {
			sa4->sin_addr.s_addr = htonl(INADDR_ANY);
			sa4->sin_port        = htons(port);
			break;
		}
		case AF_INET6: {
			::memcpy(&sa6->sin6_addr, &in6addr_any, sizeof(in6addr_any));
			sa4->sin_port = htons(port);
			break;
		}
		case AF_UNIX:   // Fall-through
		case AF_UNSPEC: // Fall-through
		default: break; // Leave as zeroes
		}
		return sas;
	}
	if( family == AF_UNIX || port < 0 ) {
		// UNIX path
		saU->sun_family = AF_UNIX;
		std::strncpy(saU->sun_path, addrstr.c_str(),
		             sizeof(saU->sun_path) - 1);
		return sas;
	}
	if( family == AF_INET || family == AF_UNSPEC ) {
		// Try IPv4 address
		if( ::inet_pton(AF_INET, addrstr.c_str(), &(sa4->sin_addr)) == 1 ) {
			sa4->sin_family = AF_INET;
			sa4->sin_port   = htons(port);
			return sas;
		}
	}
	if( family == AF_INET6 || family == AF_UNSPEC ) {
		// Try IPv6 address
		if( ::inet_pton(AF_INET6, addrstr.c_str(), &(sa6->sin6_addr)) == 1 ) {
			sa6->sin6_family = AF_INET6;
			sa6->sin6_port   = htons(port);
			return sas;
		}
	}
	// Try interface lookup
	if( Socket::addr_from_interface(addrstr.c_str(), (sockaddr*)&sas, family) ) {
		// Note: Can actually be ip4 or ip6 but the port works the same
		sa4->sin_port = htons(port);
		return sas;
	}
	// Try hostname lookup
	else if( Socket::addr_from_hostname(addrstr.c_str(), (sockaddr*)&sas, family) ) {
		// Note: Can actually be ip4 or ip6 but the port works the same
		sa4->sin_port = htons(port);
		return sas;
	}
	else {
		throw Socket::Error("Not a valid IP address, interface or hostname");
	}
}
std::string Socket::address_string(sockaddr_storage addr) {
	switch( addr.ss_family ) {
	case AF_UNIX: {
		// WAR for sun_path not always being NULL-terminated
		// TODO: Fix this up!
		/*
		char addr0[sizeof(struct sockaddr_un)+1];
		memset(addr0, 0,     sizeof(addr0));
		memcpy(addr0, &addr, sizeof(struct sockaddr_un));
		return std::string(((struct sockaddr_un*)addr0)->sun_path);
		*/
		return "<UNIMPLEMENTED>";
	}
	case AF_INET:
	case AF_INET6: {
		char buffer[INET6_ADDRSTRLEN];
		if( getnameinfo((struct sockaddr*)&addr, sizeof(addr),
		                buffer, sizeof(buffer),
		                0, 0, NI_NUMERICHOST) != 0 ) {
			return "";
		}
		else {
			return std::string(buffer);
		}
	}
	default: throw Socket::Error("Invalid address family");
	}
}
int Socket::discover_mtu(sockaddr_storage remote_address) {
	Socket s(SOCK_DGRAM);
	s.connect(remote_address);
	return s.get_option<int>(IP_MTU, IPPROTO_IP);
}
void Socket::bind(sockaddr_storage local_address,
                  int              max_conn_queue) {
	if( _mode != Socket::MODE_CLOSED ) {
		throw Socket::Error("Socket is already open");
	}
	this->open(local_address.ss_family);
	
	// Allow binding multiple sockets to one port
	//   See here for more info: https://lwn.net/Articles/542629/
	// TODO: This must be done before calling ::bind, which is slightly
	//         awkward with how this method is set up, as the user has
	//         no way to do it themselves. However, doing it by default
	//         is probably not a bad idea anyway.
#ifdef SO_REUSEPORT
	this->set_option(SO_REUSEPORT, 1);
#else
	#warning "Kernel version does not support SO_REUSEPORT; multithreaded send/recv will not be possible"
#endif
	
	check_error(::bind(_fd, (struct sockaddr*)&local_address, sizeof(local_address)),
	            "bind socket");
	if( _type == SOCK_STREAM ) {
		check_error(::listen(_fd, max_conn_queue),
		            "set socket to listen");
		_mode = Socket::MODE_LISTENING;
	}
	else {
		_mode = Socket::MODE_BOUND;
	}
}
// TODO: Add timeout support? Bit of a pain to implement.
void Socket::connect(sockaddr_storage remote_address) {
	bool can_reuse = (_fd != -1 &&
	                  _type == SOCK_DGRAM &&
	                  (remote_address.ss_family == AF_UNSPEC ||
	                   remote_address.ss_family == _family));
	if( !can_reuse ) {
		if( _mode != Socket::MODE_CLOSED ) {
			throw Socket::Error("Socket is already open");
		}
		this->open(remote_address.ss_family);
	}
	check_error(::connect(_fd, (sockaddr*)&remote_address, sizeof(remote_address)),
	            "connect socket");
	if( remote_address.ss_family == AF_UNSPEC ) {
		_mode = Socket::MODE_BOUND;
	}
	else {
		_mode = Socket::MODE_CONNECTED;
	}
}
Socket* Socket::accept(double timeout_secs) {
	if( _mode != Socket::MODE_LISTENING ) {
		throw Socket::Error("Socket is not listening");
	}
	sockaddr_storage remote_addr;
	int flags = timeout_secs >= 0 ? SOCK_NONBLOCK : 0;
	socklen_t addrsize = sizeof(remote_addr);
	int ret = ::accept4(_fd, (sockaddr*)&remote_addr, &addrsize, flags);
	if( ret == -1 && (errno == EAGAIN || errno == EWOULDBLOCK ) ) {
		pollfd pfd;
		pfd.fd      = _fd;
		pfd.events  = POLLIN | POLLERR | POLLRDNORM;
		pfd.revents = 0;
		int timeout_ms = int(std::min(timeout_secs*1e3, double(INT_MAX)) + 0.5);
		if( poll(&pfd, 1, timeout_ms) == 0 ) {
			// Timed out
			return 0;//-1;
		}
		else {
			ret = ::accept4(_fd, (sockaddr*)&remote_addr, &addrsize, flags);
			if( ret == -1 && (errno == EAGAIN || errno == EWOULDBLOCK ) ) {
				// Connection dropped before accept completed
				return 0;//-1;
			}
			else {
				check_error(ret, "accept incoming connection");
			}
		}
	}
	else {
		check_error(ret, "accept incoming connection");
	}
	//return ret; // File descriptor for new connected client socket
	return Socket::manage(ret);
}
void Socket::shutdown(int how) {
	int ret = ::shutdown(_fd, how);
	// WAR for shutdown() returning an error on unconnected DGRAM sockets, even
	//  though it still successfully unblocks recv calls in other threads,
	//  which is very useful behaviour.
	//    Note: In Python, the corresponding exception can be avoided by
	//            connecting the socket to ("0.0.0.0", 0) first.
	if( ret < 0 && errno != EOPNOTSUPP && errno != ENOTCONN ) {
		check_error(ret, "shutdown socket");
	}
}
// Note: If offsets is NULL, assumes uniform spacing of sizes[0]
void Socket::prepare_msgs(size_t            npacket,
                          void*             header_buf,
                          size_t const*     header_offsets,
                          size_t const*     header_sizes,
                          void*             payload_buf,
                          size_t const*     payload_offsets,
                          size_t const*     payload_sizes,
                          sockaddr_storage* packet_addrs) {
	mmsghdr hdr0 = {};
	_msgs.resize(npacket, hdr0);
	_iovecs.resize(npacket*2);
	for( uint64_t m=0; m<npacket; ++m ) {
		if( header_buf ) {
			size_t header_size   = (header_offsets  ?
			                        header_sizes[m] :
			                        header_sizes[0]);
			size_t header_offset = (header_offsets    ?
			                        header_offsets[m] :
			                        header_size*m);
			_iovecs[m*2+0].iov_base = &((uint8_t*)header_buf)[header_offset];
			_iovecs[m*2+0].iov_len  = header_size;
		}
		size_t payload_size   = (payload_offsets  ?
		                         payload_sizes[m] :
		                         payload_sizes[0]);
		size_t payload_offset = (payload_offsets    ?
		                         payload_offsets[m] :
		                         payload_size*m);
		_iovecs[m*2+1].iov_base = &((uint8_t*)payload_buf)[payload_offset];
		_iovecs[m*2+1].iov_len  = payload_size;
		if( header_buf ) {
			_msgs[m].msg_hdr.msg_iov     = &_iovecs[m*2+0];
			_msgs[m].msg_hdr.msg_iovlen  = 2;
		}
		else {
			_msgs[m].msg_hdr.msg_iov     = &_iovecs[m*2+1];
			_msgs[m].msg_hdr.msg_iovlen  = 1;
		}
		if( packet_addrs ) {
			_msgs[m].msg_hdr.msg_name    = (void*)&packet_addrs[m];
			_msgs[m].msg_hdr.msg_namelen = sizeof(*packet_addrs);
		}
	}
}
size_t Socket::recv_block(size_t            npacket,       // Max for UDP
                          void*             header_buf,    // Can be NULL
                          size_t const*     header_offsets,
                          size_t const*     header_sizes,
                          void*             payload_buf,
                          size_t const*     payload_offsets,
                          size_t const*     payload_sizes, // Max for UDP
                          size_t*           packet_sizes,
                          sockaddr_storage* packet_sources,
                          double            timeout_secs) {
	if( !(_mode == Socket::MODE_BOUND || _mode == Socket::MODE_CONNECTED) ) {
		throw Socket::Error("Cannot receive; not bound or connected");
	}
	// WAR for BUG in recvmmsg timeout behaviour
	if( timeout_secs > 0 ) {
		timeval timeout;
		timeout.tv_sec  = (int)timeout_secs;
		timeout.tv_usec = (int)((timeout_secs - timeout.tv_sec)*1e6);
		setsockopt(_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
	}
	// TODO: Replacing MSG_WAITFORONE with 0 results in low CPU use instead of 100%
	//         Probably add an option to the function call
	//int flags = (timeout_secs == 0) ? MSG_DONTWAIT : MSG_WAITFORONE;
	int flags = (timeout_secs == 0) ? MSG_DONTWAIT : 0;
	this->prepare_msgs(npacket,
	                   header_buf, header_offsets, header_sizes,
	                   payload_buf, payload_offsets, payload_sizes,
	                   packet_sources);
	ssize_t nmsg = recvmmsg(_fd, &_msgs[0], _msgs.size(), flags, 0);//timeout_ptr);
	if( nmsg < 0 && (errno == EAGAIN || errno == EWOULDBLOCK ) ) {
		nmsg = 0;
	}
	else {
		check_error(nmsg, "receive messages");
	}
	_nrecv_bytes = 0;
	if( packet_sizes ) {
		for( ssize_t m=0; m<nmsg; ++m ) {
			packet_sizes[m] = _msgs[m].msg_len;
			_nrecv_bytes   += _msgs[m].msg_len;
		}
	}
	// TODO: Does this actually work? (Note: The code itself is fine).
	// Check ancilliary data for dropped packet log (SO_RXQ_OVFL)
	_ndropped = 0;
	/*
	////if( ndropped && m == 0 ) {
	//if( m == 0 ) { // TODO: Which header is the drop count written to?
	for( ssize_t m=0; m<nmsg; ++m ) {
		for( cmsghdr* cmsg = CMSG_FIRSTHDR(&_msgs[m].msg_hdr);
		     cmsg != NULL;
		     cmsg = CMSG_NXTHDR(&_msgs[m].msg_hdr, cmsg)) {
			if( cmsg->cmsg_type == SO_RXQ_OVFL ) {
				unsigned* uptr = reinterpret_cast<unsigned*>(CMSG_DATA(cmsg));
				_ndropped += *uptr;
				break;
			}
		}
	}
	*/
	return nmsg;
}
size_t Socket::recv_packet(void*             header_buf,
                           size_t            header_size,
                           void*             payload_buf,
                           size_t            payload_size,
                           size_t*           packet_size,
                           sockaddr_storage* packet_source,
                           double            timeout_secs) {
	return this->recv_block(1,
	                        header_buf,  0, &header_size,
	                        payload_buf, 0, &payload_size,
	                        packet_size,
	                        packet_source,
	                        timeout_secs);
}
size_t Socket::send_block(size_t                  npacket,
                          void const*             header_buf,
                          size_t const*           header_offsets,
                          size_t const*           header_sizes,
                          void const*             payload_buf,
                          size_t const*           payload_offsets,
                          size_t const*           payload_sizes,
                          sockaddr_storage const* packet_dests, // Not needed after connect()
                          double                  timeout_secs) {
	if( !(_mode == Socket::MODE_BOUND || _mode == Socket::MODE_CONNECTED) ) {
		throw Socket::Error("Cannot send; not connected or listening");
	}
	if( packet_dests && _mode == Socket::MODE_CONNECTED ) {
		throw Socket::Error("packet_dests must be NULL for connected sockets");
	}
	else if( !packet_dests && _mode == Socket::MODE_BOUND ) {
		throw Socket::Error("packet_dests must be specified for bound sockets");
	}
	if( timeout_secs > 0 ) {
		timeval timeout;
		timeout.tv_sec  = (int)timeout_secs;
		timeout.tv_usec = (int)((timeout_secs - timeout.tv_sec)*1e6);
		setsockopt(_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
	}
	int flags = (timeout_secs == 0) ? MSG_DONTWAIT : 0;
	this->prepare_msgs(npacket,
	                   (void*)header_buf, header_offsets, header_sizes,
	                   (void*)payload_buf, payload_offsets, payload_sizes,
	                   (sockaddr_storage*)packet_dests);
	ssize_t nmsg = sendmmsg(_fd, &_msgs[0], _msgs.size(), flags);
	if( nmsg < 0 && (errno == EAGAIN || errno == EWOULDBLOCK ) ) {
		nmsg = 0;
	}
	else {
		check_error(nmsg, "send messages");
	}
	return nmsg;
}
size_t Socket::send_packet(void const*             header_buf,
                           size_t                  header_size,
                           void const*             payload_buf,
                           size_t                  payload_size,
                           sockaddr_storage const* packet_dest, // Not needed after connect()
                           double                  timeout_secs) {
	return this->send_block(1,
	                        header_buf, 0, &header_size,
	                        payload_buf, 0, &payload_size,
	                        packet_dest, timeout_secs);
}
void Socket::open(sa_family_t family) {
	this->close();
	_family = family;
	check_error(_fd = ::socket(_family, _type, 0),
	            "create socket");
	this->set_default_options();
}
void Socket::set_default_options() {
	// Increase socket buffer sizes for efficiency
	this->set_option(SO_RCVBUF, DEFAULT_SOCK_BUF_SIZE);
	this->set_option(SO_SNDBUF, DEFAULT_SOCK_BUF_SIZE);
	struct linger linger_obj;
	linger_obj.l_onoff  = 1;
	linger_obj.l_linger = DEFAULT_LINGER_SECS;
	this->set_option(SO_LINGER, linger_obj);
	// TODO: Not sure if this feature actually works
	//this->set_option(SO_RXQ_OVFL, 1); // Enable dropped packet logging
}
sockaddr_storage Socket::get_remote_address() /*const*/ {
	if( _mode != Socket::MODE_CONNECTED ) {
		throw Socket::Error("Not connected");
	}
	sockaddr_storage sas;
	socklen_t size = sizeof(sas);
	check_error(::getpeername(_fd, (sockaddr*)&sas, &size),
	            "get peer address");
	return sas;
}
sockaddr_storage Socket::get_local_address() /*const*/ {
	if( _mode != Socket::MODE_CONNECTED &&
	    _mode != Socket::MODE_BOUND ) {
		throw Socket::Error("Not bound");
	}
	sockaddr_storage sas;
	socklen_t size = sizeof(sas);
	check_error(::getsockname(_fd, (sockaddr*)&sas, &size),
	            "get socket address");
	return sas;
}
void Socket::close() {
	if( _fd >= 0 ) {
		::close(_fd);
		_fd     = -1;
		_family = AF_UNSPEC;
		_mode   = Socket::MODE_CLOSED;
	}
}
// Similar to pton(), copies first found address into *address and returns 1
//   on success, else 0.
int Socket::addr_from_hostname(const char* hostname,
                               sockaddr*   address,
                               sa_family_t family,
                               int         socktype) {
	struct addrinfo hints;
	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_family   = family;
    hints.ai_socktype = socktype;
    hints.ai_flags    = 0; // Any
    hints.ai_protocol = 0; // Any
    struct addrinfo* servinfo;
	if( ::getaddrinfo(hostname, 0, &hints, &servinfo) != 0 ) {
		return 0;
	}
	for( struct addrinfo* it=servinfo; it!=NULL; it=it->ai_next ) {
		::memcpy(address, it->ai_addr, it->ai_addrlen);
		break; // Return first address
	}
	::freeaddrinfo(servinfo);
	return 1;
}
int Socket::addr_from_interface(const char* ifname,
                                sockaddr*   address,
                                sa_family_t family) {
	ifaddrs* ifaddr;
	if( ::getifaddrs(&ifaddr) == -1 ) {
		return 0;
	}
	bool found = false;
	for( ifaddrs* ifa=ifaddr; ifa!=NULL; ifa=ifa->ifa_next ) {
		if( std::strcmp(ifa->ifa_name, ifname) != 0 ||
		    ifa->ifa_addr == NULL ) {
			continue;
		}
		sa_family_t ifa_family = ifa->ifa_addr->sa_family;
		if( (family == AF_UNSPEC && (ifa_family == AF_INET ||
		                             ifa_family == AF_INET6)) ||
		    ifa_family == family ) {
			size_t addr_size = ((ifa_family == AF_INET) ?
			                    sizeof(struct sockaddr_in) :
			                    sizeof(struct sockaddr_in6));
			::memcpy(address, ifa->ifa_addr, addr_size);
			found = true;
			break; // Return first match
		}
	}
	::freeifaddrs(ifaddr);
	return found;
}
#if __cplusplus >= 201103L
void Socket::replace(Socket& s) {
	_fd          = s._fd; s._fd = -1;
	_type        = std::move(s._type);
	_family      = std::move(s._family);
	_mode        = std::move(s._mode);
	_ndropped    = std::move(s._ndropped);
	_nrecv_bytes = std::move(s._nrecv_bytes);
	_msgs        = std::move(s._msgs);
	_iovecs      = std::move(s._iovecs);
}
#endif
void Socket::swap(Socket& s) {
	std::swap(_fd,          s._fd);
	std::swap(_type,        s._type);
	std::swap(_family,      s._family);
	std::swap(_mode,        s._mode);
	std::swap(_ndropped,    s._ndropped);
	std::swap(_nrecv_bytes, s._nrecv_bytes);
	std::swap(_msgs,        s._msgs);
	std::swap(_iovecs,      s._iovecs);
}
Socket::Socket(int fd, ManageTag ) : _fd(fd) {
	_type   = this->get_option<sock_type>(SO_TYPE);
	_family = this->get_option<int>(SO_DOMAIN);
	if( this->get_option<int>(SO_ACCEPTCONN) ) {
		_mode = Socket::MODE_LISTENING;
	}
	else {
		// Not listening
		try {
			_mode = Socket::MODE_CONNECTED;
			this->get_remote_address();
		}
		catch( Socket::Error const& ) {
			// Not connected
			try {
				_mode = Socket::MODE_BOUND;
				this->get_local_address();
			}
			catch( Socket::Error const& ) {
				// Not bound
				_mode = Socket::MODE_CLOSED;
			}
		}
	}
	this->set_default_options();
}
