/*
 * Copyright (c) 2020, The Bifrost Authors. All rights reserved.
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

#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <linux/if_ether.h>

#include <infiniband/verbs.h>

#ifndef BF_VERBS_NQP
#define BF_VERBS_NQP 1
#endif

#ifndef BF_VERBS_NPKTBUF
#define BF_VERBS_NPKTBUF 32768
#endif

#ifndef BF_VERBS_WCBATCH
#define BF_VERBS_WCBATCH 16
#endif

#define BF_VERBS_PAYLOAD_OFFSET 42

struct bf_ibv_recv_pkt {
    struct ibv_recv_wr wr;
    uint32_t length;
};

// Structure for defining various types of flow rules
struct bf_ibv_flow {
  struct ibv_flow_attr         attr;
  struct ibv_flow_spec_eth     spec_eth;
  struct ibv_flow_spec_ipv4    spec_ipv4;
  struct ibv_flow_spec_tcp_udp spec_tcp_udp;
} __attribute__((packed));

class Verbs {
    int                      _fd;
    size_t                   _pkt_size_max;
    
    struct ibv_device**      _dev_list = NULL;
    struct ibv_context*      _ctx = NULL;
    struct ibv_device_attr   _dev_attr;
    uint8_t                  _port_num;
    struct ibv_pd*           _pd = NULL;
    struct ibv_comp_channel* _cc = NULL; 
    struct ibv_cq**          _cq = NULL;
    struct ibv_qp**          _qp = NULL;
    struct ibv_sge*          _sge = NULL;
    struct ibv_flow**        _flows = NULL;
    
    struct bf_ibv_recv_pkt*  _pkt_buf = NULL;
    struct bf_ibv_recv_pkt*  _pkt = NULL;
    struct bf_ibv_recv_pkt*  _chain = NULL;
    
    uint8_t*                 _mr_buf = NULL;
    size_t                   _mr_size = 0;
    struct ibv_mr*           _mr = NULL;
    
    int32_t                  _nqp = BF_VERBS_NQP;
    int32_t                  _npkt = BF_VERBS_NPKTBUF;
    int32_t                  _nflows = 1;
    
    void get_interface_name(char* name) {
        struct sockaddr_in sin;
        char ip[INET_ADDRSTRLEN];
        socklen_t len = sizeof(sin);
        check_error(::getsockname(_fd, (struct sockaddr *)&sin, &len),
                    "query socket name");
        inet_ntop(AF_INET, &(sin.sin_addr), ip, INET_ADDRSTRLEN);
        
        // TODO: Is there a better way to find this?
        char cmd[256] = {'\0'};
        char line[256] = {'\0'};
        int is_lo = 0;
        sprintf(cmd, "ip route get to %s | grep dev | awk '{print $4}'", ip);
        FILE* fp = popen(cmd, "r");
        if( fgets(line, sizeof(line), fp) != NULL) {
            if( line[strlen(line)-1] == '\n' ) {
                line[strlen(line)-1] = '\0';
            }
            if( strncmp(&(line[0]), "lo", 2) == 0 ) {
                is_lo = 1;
            }
            strncpy(name, &(line[0]), IFNAMSIZ);
        }
        pclose(fp);
        
        if( is_lo ) {
            // TODO: Is there a way to avoid having to do this?
            sprintf(cmd, "ip route show | grep %s | grep -v default | awk '{print $3}'", ip);
            fp = popen(cmd, "r");
            if( fgets(line, sizeof(line), fp) != NULL) {
                if( line[strlen(line)-1] == '\n' ) {
                    line[strlen(line)-1] = '\0';
                } 
                strncpy(name, &(line[0]), IFNAMSIZ);
            }
            pclose(fp);
        }
    }
    void get_mac_address(uint8_t* mac) {
        struct ifreq ethreq;
        this->get_interface_name(&(ethreq.ifr_name[0]));
        check_error(::ioctl(_fd, SIOCGIFHWADDR, &ethreq),
                    "query interface hardware address");
        
        ::memcpy(mac, (uint8_t*) ethreq.ifr_hwaddr.sa_data, 6);
    }
    uint16_t get_port() {
        struct sockaddr_in sin;
        socklen_t len = sizeof(sin);
        check_error(::getsockname(_fd, (struct sockaddr *)&sin, &len),
                    "query socket name");
        return ntohs(sin.sin_port);
    }
    uint64_t get_interface_gid() {
        uint64_t id;
        uint8_t mac[6] = {0};
        uint8_t buf[8] = {0};
        this->get_mac_address(&(mac[0]));
        
        ::memcpy(buf, (unsigned char*) &(mac[0]), 3);
        buf[0] ^= 2; // Toggle G/L bit per modified EUI-64 spec
        buf[3] = 0xff;
        buf[4] = 0xfe;
        ::memcpy(buf+5, (unsigned char*)  &(mac[3]), 3);
        ::memcpy(&id, buf, 8);
        return id;
    }
    void create_context();
    void destroy_context();
    void create_buffers();
    void destroy_buffers();
    void create_queues();
    void destroy_queues();
    void link_work_requests();
    void create_flows();
    void destroy_flows();
    int release(bf_ibv_recv_pkt*);
    struct bf_ibv_recv_pkt* receive(int timeout_ms=1);
    inline void check_error(int retval, std::string what) {
        if( retval < 0 ) {
            destroy_flows();
            destroy_queues();
            destroy_buffers();
            destroy_context();
            
            std::stringstream ss;
            ss << "Failed to " << what << ": (" << errno << ") "
               << strerror(errno);
            throw Verbs::Error(ss.str());
        }
    }
    inline void check_null(void* ptr, std::string what) {
        if( ptr == NULL ) {
            destroy_flows();
            destroy_queues();
            destroy_buffers();
            destroy_context();
            
            std::stringstream ss;
            ss << "Failed to " << what;
            throw Verbs::Error(ss.str());
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
    
    Verbs(int fd, size_t pkt_size_max)
        : _fd(fd), _pkt_size_max(pkt_size_max) {
            create_context();
            create_buffers();
            create_queues();
            link_work_requests();
            create_flows();
    }
    ~Verbs() {
        destroy_flows();
        destroy_queues();
        destroy_buffers();
        destroy_context();
    }
    inline int recv_packet(uint8_t* buf, size_t bufsize, uint8_t** pkt_ptr, int flags=0) {
        // If we don't have a work-request queue on the go,
        // get some new packets.
        if( _pkt ) {
            _pkt = (struct bf_ibv_recv_pkt *)_pkt->wr.next;
            if( !_pkt ) {
                release(_chain);
                _chain = NULL;
            }
        }
        while( !_chain ) {
            _chain = receive(1);
            _pkt = _chain;
        }
        // IBV returns Eth/UDP/IP headers. Strip them off here.
        *pkt_ptr = (uint8_t *)_pkt->wr.sg_list->addr + BF_VERBS_PAYLOAD_OFFSET;
        return _pkt->length - BF_VERBS_PAYLOAD_OFFSET;
    }
};

