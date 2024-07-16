/*
 * Copyright (c) 2022, The Bifrost Authors. All rights reserved.
 * Copyright (c) 2022, The University of New Mexico. All rights reserved.
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
#include <bifrost/config.h>
#include <bifrost/rdma.h>

#include <arpa/inet.h>  // For ntohs
#include <sys/mman.h>   // For mlock

#include <stdexcept>
#include <string>
#include <sstream>
#include <memory>
#include <cstdlib>      // For posix_memalign
#include <cstring>      // For memcpy, memset

#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>

struct __attribute__((aligned(8))) bf_message {
  enum {
    MSG_HEADER = 0,
    MSG_SPAN   = 1,
    MSG_REPLY  = 2,
  } type;
  uint64_t time_tag;
  uint64_t offset;
  uint64_t length;
};

struct bf_rdma {
  struct rdma_cm_id* listen_id;
  struct rdma_cm_id* id;
  
  struct rdma_addrinfo* res;
  struct ibv_wc wc;
  
  size_t buf_size;
  uint8_t* buf;
  uint8_t* send_buf;
  struct ibv_mr* mr;
  struct ibv_mr* send_mr;
};

class Rdma {
  int     _fd;
  size_t  _message_size;
  bool _is_server;
  bf_rdma _rdma;
  
  void get_ip_address(char* ip) {
    sockaddr_in sin;
    socklen_t len = sizeof(sin);
    if( _is_server ) {
      check_error(::getsockname(_fd, (sockaddr *)&sin, &len),
                  "query socket name");
    } else {
      check_error(::getpeername(_fd, (sockaddr *)&sin, &len),
                  "query peer name");
    }
    inet_ntop(AF_INET, &(sin.sin_addr), ip, INET_ADDRSTRLEN);
  }
  uint16_t get_port() {
    sockaddr_in sin;
    socklen_t len = sizeof(sin);
    if( _is_server ) {
      check_error(::getsockname(_fd, (sockaddr *)&sin, &len),
                  "query socket name");
    } else {
      check_error(::getpeername(_fd, (sockaddr *)&sin, &len),
                  "query peer name");
    }
    return ntohs(sin.sin_port);
  }
  void create_connection(size_t message_size) {
    struct rdma_addrinfo hints;
  	struct ibv_qp_init_attr init_attr;
    struct ibv_qp_attr qp_attr;
  	
    char server[INET_ADDRSTRLEN];
    char port[7];
    this->get_ip_address(&(server[0]));
    snprintf(&(port[0]), 7, "%d", this->get_port());
    
  	::memset(&hints, 0, sizeof(hints));
    if( _is_server ) {
      hints.ai_flags = RAI_PASSIVE;
    }
  	hints.ai_port_space = RDMA_PS_TCP;
  	check_error(rdma_getaddrinfo(&(server[0]), &(port[0]), &hints, &(_rdma.res)),
                "query RDMA address information");
    
    ::memset(&init_attr, 0, sizeof(init_attr));
  	init_attr.cap.max_send_wr = init_attr.cap.max_recv_wr = 1;
  	init_attr.cap.max_send_sge = init_attr.cap.max_recv_sge = 1;
  	init_attr.cap.max_inline_data = 0;
    if( !_is_server ) {
      init_attr.qp_context = _rdma.id;
    }
  	init_attr.sq_sig_all = 1;
    
    if( _is_server ) {
      check_error(rdma_create_ep(&(_rdma.listen_id), _rdma.res, NULL, &init_attr),
                  "create the RDMA identifier");
      
      check_error(rdma_listen(_rdma.listen_id, 0),
                  "listen for incoming connections");
      
      check_error(rdma_get_request(_rdma.listen_id, &(_rdma.id)),
                  "get connection request");
      
      ::memset(&qp_attr, 0, sizeof(qp_attr));
    	::memset(&init_attr, 0, sizeof(init_attr));
    	check_error(ibv_query_qp(_rdma.id->qp, &qp_attr, IBV_QP_CAP, &init_attr),
                  "query the queue pair");
    } else {
      check_error(rdma_create_ep(&(_rdma.id), _rdma.res, NULL, &init_attr),
                  "create the RDMA identifier");
    }
    
    _rdma.buf_size = message_size + sizeof(bf_message);
    check_error(::posix_memalign((void**)&(_rdma.buf), 32, _rdma.buf_size),
                "allocate buffer");
    check_error(::mlock(_rdma.buf, _rdma.buf_size),
                "lock buffer");
    _rdma.mr = rdma_reg_msgs(_rdma.id, _rdma.buf, _rdma.buf_size);
    check_null(_rdma.mr, "create memory region");
    
    check_error(::posix_memalign((void**)&(_rdma.send_buf), 32, _rdma.buf_size),
                "allocate send buffer");
    check_error(::mlock(_rdma.send_buf, _rdma.buf_size),
                "lock send buffer");
    _rdma.send_mr = rdma_reg_msgs(_rdma.id, _rdma.send_buf, _rdma.buf_size);
    check_null(_rdma.send_mr, "create send memory region");
    
    if( _is_server ) {
      check_error(rdma_accept(_rdma.id, NULL),
                  "accept incomining connections");
    } else {
        check_error(rdma_connect(_rdma.id, NULL),
                  "connect");
        
        check_error(rdma_post_recv(_rdma.id, NULL, _rdma.buf, _rdma.buf_size, _rdma.mr),
                    "set RDMA post receive");
    }
  }
  inline void close_connection() {
    if( _rdma.id ) {
      rdma_disconnect(_rdma.id);
    }
    
    if( _rdma.mr ) {
      rdma_dereg_mr(_rdma.mr);
    }
    
    if( _rdma.send_mr ) {
      rdma_dereg_mr(_rdma.send_mr);
    }
    
    if( _rdma.buf ) {
      ::munlock(_rdma.buf, _rdma.buf_size);
      ::free(_rdma.buf);
    }
    
    if( _rdma.send_buf ) {
      ::munlock(_rdma.send_buf, _rdma.buf_size);
      ::free(_rdma.send_buf);
    }
    
    if( _rdma.id ) {
      rdma_destroy_ep(_rdma.id);
    }
    
    if( _rdma.listen_id ) {
      rdma_destroy_ep(_rdma.listen_id);
    }
    
    if( _rdma.res ) {
      rdma_freeaddrinfo(_rdma.res);
    }
  }
  inline void check_error(int retval, std::string what) {
    if( retval < 0 ) {
      close_connection();
      
      std::stringstream ss;
      ss << "Failed to " << what << ": (" << errno << ") "
         << strerror(errno);
      throw Rdma::Error(ss.str());
    }
  }
  inline void check_null(void* ptr, std::string what) {
    if( ptr == NULL ) {
      close_connection();
      
      std::stringstream ss;
      ss << "Failed to " << what << ": (" << errno << ") "
         << strerror(errno);
      throw Rdma::Error(ss.str());
    }
  }
public:
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
  
  Rdma(int fd, size_t message_size, bool is_server)
    : _fd(fd), _message_size(message_size), _is_server(is_server) {
      ::memset(&_rdma, 0, sizeof(_rdma));
      create_connection(message_size);
  }
  ~Rdma() {
    close_connection();
  }
  inline size_t max_message_size() { return _message_size; }
  inline ssize_t send(const bf_message* msg, const void* buf) {
    ::memcpy(_rdma.send_buf, msg, sizeof(bf_message));
    ::memcpy(_rdma.send_buf+sizeof(bf_message), buf, msg->length);
    
    check_error(rdma_post_send(_rdma.id, NULL, _rdma.send_buf, _rdma.buf_size, _rdma.send_mr, 0),
                "queue send request");
    check_error(rdma_get_send_comp(_rdma.id, &(_rdma.wc)),
                "get send completion");
    
    bf_message *reply;
    check_error(rdma_post_recv(_rdma.id, NULL, _rdma.buf, sizeof(bf_message), _rdma.mr),
                "queue receive request");
    check_error(rdma_get_recv_comp(_rdma.id, &(_rdma.wc)),
                "get receive completion");
    
    reply = reinterpret_cast<bf_message*>(_rdma.buf);
    if( reply->type != bf_message::MSG_REPLY ) {
      return -1;
    }
    return msg->length;
  }
  inline ssize_t receive(bf_message* msg, void* buf) {
    check_error(rdma_get_recv_comp(_rdma.id, &(_rdma.wc)),
                "get receive completion");
    
    ::memcpy(msg, _rdma.buf, sizeof(bf_message));
    ::memcpy(buf, _rdma.buf+sizeof(bf_message), msg->length);
    
    bf_message* reply;
    reply = reinterpret_cast<bf_message*>(_rdma.send_buf);
    reply->type = bf_message::MSG_REPLY;
    check_error(rdma_post_send(_rdma.id, NULL, _rdma.send_buf, sizeof(bf_message), _rdma.send_mr, 0),
                "queue send request");
    check_error(rdma_get_send_comp(_rdma.id, &(_rdma.wc)),
               "get send completion");
    check_error(rdma_post_recv(_rdma.id, NULL, _rdma.buf, _rdma.buf_size, _rdma.mr),
                "queue receive request");
    
    return msg->length;
  }
};

class BFrdma_impl {
  Rdma* _rdma;
  
public:
  inline BFrdma_impl(int fd, size_t message_size, int is_server) {
    _rdma = new Rdma(fd, message_size, is_server);
  }
  inline size_t max_message_size() { return _rdma->max_message_size(); }
  ssize_t send_header(BFoffset    time_tag,
                      BFsize      header_size,
                      const void* header,
                      BFoffset    offset_from_head);
  ssize_t send_span(BFarray const* span);
  ssize_t receive(BFoffset* time_tag,
                  BFsize*   header_size,
                  BFoffset* offset_from_head,
                  BFsize*   span_size,
                  void*     contents);
};
