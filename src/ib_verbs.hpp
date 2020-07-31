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

#include "hw_locality.hpp"

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
#include <netinet/if_ether.h>
#include <sys/mman.h>

extern "C" {
#include <infiniband/verbs.h>
}

#ifndef BF_VERBS_NQP
#define BF_VERBS_NQP 1
#endif

#ifndef BF_VERBS_NPKTBUF
#define BF_VERBS_NPKTBUF 8192
#endif

#ifndef BF_VERBS_WCBATCH
#define BF_VERBS_WCBATCH 16
#endif

#define BF_VERBS_PAYLOAD_OFFSET 42

extern "C" {

struct bf_ibv_recv_pkt{
    ibv_recv_wr wr;
    uint32_t    length;
    uint64_t    timestamp;
};

struct bf_ibv_flow {
    ibv_flow_attr         attr;
    ibv_flow_spec_eth     eth;
    ibv_flow_spec_ipv4    ipv4;
    ibv_flow_spec_tcp_udp udp;
} __attribute__((packed));

struct bf_ibv {
    ibv_context*      ctx;
    //ibv_device_attr   dev_attr;
    uint8_t           port_num;
    ibv_pd*           pd;
    ibv_comp_channel* cc; 
    ibv_cq**          cq;
    ibv_qp**          qp;
    ibv_sge*          sge;
    ibv_flow**        flows;
    
    uint8_t*          mr_buf;
    size_t            mr_size;
    ibv_mr*           mr;
    
    bf_ibv_recv_pkt*  pkt_buf;
    bf_ibv_recv_pkt*  pkt;
    bf_ibv_recv_pkt*  pkt_batch;
} __attribute((aligned(8)));

} // extern "C"

class Verbs : public BoundThread {
    int    _fd;
    size_t _pkt_size_max;
    int    _timeout;
    bf_ibv _verbs;
    
    void get_interface_name(char* name) {
        sockaddr_in sin;
        char ip[INET_ADDRSTRLEN];
        socklen_t len = sizeof(sin);
        check_error(::getsockname(_fd, (sockaddr *)&sin, &len),
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
        ifreq ethreq;
        this->get_interface_name(&(ethreq.ifr_name[0]));
        check_error(::ioctl(_fd, SIOCGIFHWADDR, &ethreq),
                    "query interface hardware address");
        
        ::memcpy(mac, (uint8_t*) ethreq.ifr_hwaddr.sa_data, 6);
    }
    void get_ip_address(char* ip) {
        sockaddr_in sin;
        socklen_t len = sizeof(sin);
        check_error(::getsockname(_fd, (sockaddr *)&sin, &len),
                    "query socket name");
        inet_ntop(AF_INET, &(sin.sin_addr), ip, INET_ADDRSTRLEN);
    }
    uint16_t get_port() {
        sockaddr_in sin;
        socklen_t len = sizeof(sin);
        check_error(::getsockname(_fd, (sockaddr *)&sin, &len),
                    "query socket name");
        return ntohs(sin.sin_port);
    }
    int get_timeout_ms() {
        timeval value;
        socklen_t size = sizeof(value);
        check_error(::getsockopt(_fd, SOL_SOCKET, SO_RCVTIMEO, &value, &size),
                    "query socket timeout");
        return int(value.tv_sec*1000) + int(value.tv_usec/1000);
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
    void create_buffers(size_t pkt_size_max);
    void destroy_buffers();
    void create_queues();
    void destroy_queues();
    void link_work_requests(size_t pkt_size_max);
    void create_flows();
    void destroy_flows();
    int release(bf_ibv_recv_pkt*);
    bf_ibv_recv_pkt* receive(int timeout_ms=1);
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
            ss << "Failed to " << what << ": (" << errno << ") "
               << strerror(errno);
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
    
    Verbs(int fd, size_t pkt_size_max, int core)
        : BoundThread(core), _fd(fd), _pkt_size_max(pkt_size_max) {
            ::memset(&_verbs, 0, sizeof(_verbs));
            _timeout = get_timeout_ms();
            
            create_context();
            create_buffers(pkt_size_max);
            create_queues();
            link_work_requests(pkt_size_max);
            create_flows();
    }
    ~Verbs() {
        destroy_flows();
        destroy_queues();
        destroy_buffers();
        destroy_context();
    }
    int recv_packet(uint8_t** pkt_ptr, int flags=0);
};

