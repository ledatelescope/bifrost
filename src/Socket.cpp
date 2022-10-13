/*                                               -*- indent-tabs-mode:nil -*-
 * Copyright (c) 2022, The Bifrost Authors. All rights reserved.
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

#include "Socket.hpp"

#if defined __APPLE__ && __APPLE__

static int accept4(int sockfd,
                   struct sockaddr *addr,
                   socklen_t *addrlen,
                   int flags) {
  return ::accept(sockfd, addr, addrlen);
}

static sa_family_t get_family(int sockfd) {
  sockaddr addr;
  socklen_t addr_len = sizeof(addr);
  sockaddr_in* addr4 = reinterpret_cast<sockaddr_in*>(&addr);
  if( ::getsockname(sockfd, (struct sockaddr*)&addr, &addr_len) < 0 ) {
    return AF_UNSPEC;
  }
  return addr4->sin_family;
}

static int get_mtu(int sockfd) {
  int mtu = 0;
  sa_family_t family = ::get_family(sockfd);

  sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  sockaddr_in*  addr4 = reinterpret_cast<sockaddr_in*> (&addr);
  sockaddr_in6* addr6 = reinterpret_cast<sockaddr_in6*>(&addr);
  ::getsockname(sockfd, (struct sockaddr*)&addr, &addr_len);

  ifaddrs* ifaddr;
  if( ::getifaddrs(&ifaddr) == -1 ) {
    return 0;
  }

  ifreq ifr;
  bool found = false;
  for( ifaddrs* ifa=ifaddr; ifa!=NULL; ifa=ifa->ifa_next ) {
    if( ifa->ifa_addr == NULL || found) {
      continue;
    }
    sa_family_t ifa_family = ifa->ifa_addr->sa_family;
    if( (family == AF_UNSPEC && (ifa_family == AF_INET ||
                                 ifa_family == AF_INET6)) ||
        ifa_family == family ) {
      if( ifa_family == AF_INET ) {
        struct sockaddr_in* inaddr = (struct sockaddr_in*) ifa->ifa_addr;
        if( inaddr->sin_addr.s_addr == addr4->sin_addr.s_addr ) {
          found = true;
        }
      } else if( ifa_family == AF_INET6 ) {
        struct sockaddr_in6* inaddr6 = (struct sockaddr_in6*) ifa->ifa_addr;
        if( std::memcmp(inaddr6->sin6_addr.s6_addr, addr6->sin6_addr.s6_addr, 16) == 0 ) {
          found = true;
        }
      }
    }

    if( found ) {
      ::strncpy(ifr.ifr_name, ifa->ifa_name, IFNAMSIZ-1);
      ifr.ifr_name[IFNAMSIZ-1] = '\0';
      if( ::ioctl(sockfd, SIOCGIFMTU, &ifr) != -1) {
        mtu = ifr.ifr_mtu;
      }
    }
  }
  ::freeifaddrs(ifaddr);

  return mtu;
}

// TODO: What about recvmsg_x?
int recvmmsg(int sockfd,
             struct mmsghdr *msgvec,
             unsigned int vlen,
             int flags,
             struct timespec *timeout) {
  int count = 0;
  int recv;
  for(unsigned int i=0; i<vlen; i++) {
    recv = ::recvmsg(sockfd, &(msgvec[count].msg_hdr), flags);
    if(recv > 0) {
      count++;
    }
  }
  return count;
}

// TODO: What about sendmsg_x?
int sendmmsg(int sockfd,
             struct mmsghdr *msgvec,
             unsigned int vlen,
             int flags) {
  int count = 0;
  int sent;
  for(unsigned int i=0; i<vlen; i++) {
    sent = ::sendmsg(sockfd, &(msgvec[i].msg_hdr), flags);
    msgvec[i].msg_len = sent;
    count++;
  }
  return count;
}

#endif // __APPLE__

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

std::string Socket::address_string(sockaddr_storage const& addr) {
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

int Socket::discover_mtu(sockaddr_storage const& remote_address) {
  Socket s(SOCK_DGRAM);
  s.connect(remote_address);
#if defined __APPLE__ && __APPLE__
  return ::get_mtu(s.get_fd());
#else
  return s.get_option<int>(IP_MTU, IPPROTO_IP);
#endif
}

void Socket::bind(sockaddr_storage const& local_address,
                  int                     max_conn_queue) {
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
void Socket::connect(sockaddr_storage const& remote_address) {
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
  check_error(::connect(_fd, (sockaddr*)&remote_address, sizeof(sockaddr)),
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

int Socket::get_mtu() /*const*/ {
  if( _mode != Socket::MODE_CONNECTED ) {
    throw Socket::Error("Not connected");
  }
#if defined __APPLE__ && __APPLE__
  return ::get_mtu(_fd);
#else
  return this->get_option<int>(IP_MTU, IPPROTO_IP);
#endif
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
  _type   = this->get_option<int>(SO_TYPE);
#if defined __APPLE__ && __APPLE__
  _family = get_family(fd);
#else
  _family = this->get_option<int>(SO_DOMAIN);
#endif
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
