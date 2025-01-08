/*
 * Copyright (c) 2016-2022, The Bifrost Authors. All rights reserved.
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

#if defined __linux__ && __linux__

//#include <linux/net.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>

#endif


#if defined __APPLE__ && __APPLE__

#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>

#define SOCK_NONBLOCK O_NONBLOCK

typedef struct mmsghdr {
       struct msghdr msg_hdr;  /* Message header */
       unsigned int  msg_len;  /* Number of bytes transmitted */
} mmsghdr;

int recvmmsg(int sockfd,
                           struct mmsghdr *msgvec,
                           unsigned int vlen,
                           int flags,
                           struct timespec *timeout);
int sendmmsg(int sockfd,
                           struct mmsghdr *msgvec,
                           unsigned int vlen,
                           int flags);
#endif // __APPLE__

class Socket {
	// Not copy-assignable
	Socket(Socket const& );
	Socket& operator=(Socket const& );
	void replace(Socket& s);
	// Manage an existing socket descriptor
	// Note: Accessible only via the named constructor Socket::manage
	struct ManageTag {};
	Socket(int fd, ManageTag );
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
#if defined __APPLE__ && __APPLE__
    DEFAULT_SOCK_BUF_SIZE  = 4*1024*1024,
		DEFAULT_LINGER_SECS    = 1,
#else
		DEFAULT_SOCK_BUF_SIZE  = 256*1024*1024,
		DEFAULT_LINGER_SECS    = 3,
#endif
		DEFAULT_MAX_CONN_QUEUE = 128
	};
	// Manage an existing socket (usually one returned by Socket::accept())
	// TODO: With C++11 this could return by value (moved), which would be nicer
	static Socket* manage(int fd) { return new Socket(fd, ManageTag()); }
	explicit       Socket(int type=SOCK_DGRAM)
		: _fd(-1), _type(type), _family(AF_UNSPEC),
		  _mode(Socket::MODE_CLOSED) {
		if( !(type == SOCK_DGRAM || type == SOCK_STREAM) ) {
		  throw Socket::Error("Invalid socket type");
		}
	}
	
	virtual ~Socket() { this->close(); }
	// Move semantics
	Socket(Socket&& s)                           { this->replace(s); }
	Socket& operator=(Socket&& s) { this->close(); this->replace(s); return *this; }
	void swap(Socket& s);
	// Address generator
	// Note: Supports UNIX paths, IPv4 and IPv6 addrs, interfaces and hostnames
	//       Passing addr=0  means "any address"
	//       Passing port=-1 implies family=AF_UNIX
	static sockaddr_storage address(std::string    addr,
	                                // Note: int so that -1 can be given
	                                int            port,
	                                sa_family_t    family=AF_UNSPEC);
	static sockaddr_storage any_address(sa_family_t family=AF_UNSPEC);
	static std::string      address_string(sockaddr_storage const& addr);
	static int              discover_mtu(sockaddr_storage const& remote_address);
	
	// Server initialisation
	void bind(sockaddr_storage& local_address,
	          int              max_conn_queue=DEFAULT_MAX_CONN_QUEUE);
	void sniff(sockaddr_storage const& local_address,
	          int              max_conn_queue=DEFAULT_MAX_CONN_QUEUE);
	// Client initialisation
	void connect(sockaddr_storage const& remote_address);
	// Accept incoming SOCK_STREAM connection requests
	// TODO: With C++11 this could return by value (moved), which would be nicer
	Socket* accept(double timeout_secs=-1);
	// Note: This can be used to unblock recv calls from another thread
	//         This behaviour is not explicitly documented, but it works, is
	//           much simpler than having to mess around with select/poll, and
	//           is better than relying on timeouts.
	//           IMHO this should be official behaviour of POSIX shutdown!
	void shutdown(int how=SHUT_RD);
	//void shutdown(int how=SHUT_RDWR);
	void close();
	// Send/receive
	// Note: These four methods return the number of packets received/sent
	size_t recv_block(size_t            npacket,       // Max for UDP
	                  void*             header_buf,    // Can be NULL
	                  size_t const*     header_offsets,
	                  size_t const*     header_sizes,
	                  void*             payload_buf,
	                  size_t const*     payload_offsets,
	                  size_t const*     payload_sizes, // Max for UDP
	                  size_t*           packet_sizes,
	                  sockaddr_storage* packet_sources=0,
	                  double            timeout_secs=-1);
	size_t recv_packet(void*             header_buf,
	                   size_t            header_size,
	                   void*             payload_buf,
	                   size_t            payload_size,
	                   size_t*           packet_size,
	                   sockaddr_storage* packet_source=0,
	                   double            timeout_secs=-1);
	// No. dropped packets detected during last call to recv_*
	size_t get_drop_count() const { return _ndropped; }
	// No. bytes received by last call to recv_*
	// Note: Only valid if packet_sizes was non-NULL, otherwise returns 0
	size_t get_recv_size()  const { return _nrecv_bytes; }
	size_t send_block(size_t                  npacket,
	                  void   const*           header_buf,
	                  size_t const*           header_offsets,
	                  size_t const*           header_sizes,
	                  void   const*           payload_buf,
	                  size_t const*           payload_offsets,
	                  size_t const*           payload_sizes,
	                  sockaddr_storage const* packet_dests=0, // Not needed after connect()
	                  double                  timeout_secs=-1);
	size_t send_packet(void const*             header_buf,
	                   size_t                  header_size,
	                   void const*             payload_buf,
	                   size_t                  payload_size,
	                   sockaddr_storage const* packet_dest=0, // Not needed after connect()
	                   double                  timeout_secs=-1);
	sockaddr_storage get_local_address()  /*const*/; // check_error is non-const
	sockaddr_storage get_remote_address() /*const*/;
	int get_mtu() /*const*/;
	template<typename T>
	void set_option(int optname, T value, int level=SOL_SOCKET) {
		//::setsockopt(_fd, level, optname, &value, sizeof(value));
		check_error( ::setsockopt(_fd, level, optname, &value, sizeof(value)),
		             "set socket option" );
	}
	// Note: non-const because check_error closes the socket on failure
	template<typename T>
	T get_option(int optname, int level=SOL_SOCKET) /*const*/ {
		T value;
		socklen_t size = sizeof(value);
		check_error( ::getsockopt(_fd, level, optname, &value, &size),
		             "get socket option");
		return value;
	}
	int get_fd() const { return _fd; }
	void set_timeout(double secs) {
		if( secs > 0 ) {
			timeval timeout;
			timeout.tv_sec  = (int)secs;
			timeout.tv_usec = (int)((secs - timeout.tv_sec)*1e6);
			this->set_option(SO_RCVTIMEO, timeout);
			this->set_option(SO_SNDTIMEO, timeout);
		}
	}
	double get_timeout() const {
		// WAR for non-const get_option (which is because of close-on-error)
		Socket* self = const_cast<Socket*>(this);
		// TODO: This ignores SO_SNDTIMEO
		timeval timeout = self->get_option<timeval>(SO_RCVTIMEO);
		return timeout.tv_sec + timeout.tv_usec*1e-6;
	}
	void set_promiscuous(int state);
	int get_promiscuous();
	
private:
	void open(sa_family_t family, int protocol=0);
	void set_default_options();
	void check_error(int retval, std::string what) {
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
	void prepare_msgs(size_t            npacket,
	                  void*             header_buf,
	                  size_t const*     header_offsets,
	                  size_t const*     header_sizes,
	                  void*             payload_buf,
	                  size_t const*     payload_offsets,
	                  size_t const*     payload_sizes,
	                  sockaddr_storage* packet_addrs);
	static int addr_from_hostname(const char* hostname,
	                              sockaddr*   address,
	                              sa_family_t family=AF_UNSPEC,
	                              int         socktype=0);
	static int addr_from_interface(const char* ifname,
	                               sockaddr*   address,
	                               sa_family_t family=AF_UNSPEC);
	static int interface_from_addr(sockaddr*   address,
	                               char* ifname,
	                               sa_family_t family=AF_UNSPEC);
	                                                              
	int         _fd;
	int         _type;
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
