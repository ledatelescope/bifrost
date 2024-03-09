/*
 * Copyright (c) 2020-2023, The Bifrost Authors. All rights reserved.
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
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <linux/if_ether.h>
#include <net/if_arp.h>
#include <netinet/if_ether.h>
#include <ifaddrs.h>
#include <sys/mman.h>
#include <errno.h>

#include <infiniband/verbs.h>

// Catch for older InfiniBand verbs installs that do not have the
//#define BF_ENABLE_VERBS_OFFLOAD 0

#ifndef BF_VERBS_SEND_NQP
#define BF_VERBS_SEND_NQP 1
#endif

#ifndef BF_VERBS_SEND_NPKTBUF
#define BF_VERBS_SEND_NPKTBUF 512
#endif

#ifndef BF_VERBS_SEND_WCBATCH
#define BF_VERBS_SEND_WCBATCH 16
#endif

#ifndef BF_VERBS_SEND_NPKTBURST
#define BF_VERBS_SEND_NPKTBURST 16
#endif

#ifndef BF_VERBS_SEND_PACING
#define BF_VERBS_SEND_PACING 0
#endif

#define BF_VERBS_SEND_PAYLOAD_OFFSET 42

struct bf_ibv_send_pkt{
    ibv_send_wr wr;
    ibv_sge     sg;
};

struct bf_ibv_send {
    ibv_context*      ctx;
    uint8_t           port_num;
    ibv_pd*           pd;
    ibv_comp_channel* cc; 
    ibv_cq**          cq;
    ibv_qp**          qp;
    
    uint8_t*          mr_buf;
    size_t            mr_size;
    ibv_mr*           mr;
    
    bf_ibv_send_pkt*  pkt_buf;
    bf_ibv_send_pkt*  pkt_head;
    
    int               nqueued;
    
    uint8_t           offload_csum;
    uint32_t          hardware_pacing[2];
};

struct __attribute__((packed)) bf_ethernet_hdr {
    uint8_t dst_mac[6];
    uint8_t src_mac[6];
    uint16_t type;
};

struct __attribute__((packed)) bf_ipv4_hdr {
    uint8_t version_ihl;
    uint8_t tos;
    uint16_t length;
    uint16_t id;
    uint16_t flags_frag;
    uint8_t ttl;
    uint8_t proto;
    uint16_t checksum;
    uint32_t src_addr;
    uint32_t dst_addr;
};

inline void bf_ipv4_update_checksum(bf_ipv4_hdr* hdr) {
  hdr->checksum = 0;
  uint16_t *block = reinterpret_cast<uint16_t*>(hdr);
  
  uint32_t checksum = 0;
  for(uint32_t i=0; i<sizeof(bf_ipv4_hdr)/2; i++) {
      checksum += ntohs(*(block + i));
  }
  while( checksum > 0xFFFF ) {
      checksum = (checksum & 0xFFFF) + (checksum >> 16);
  }
  hdr->checksum = ~htons((uint16_t) checksum);
}

struct __attribute__((packed)) bf_udp_hdr {
   uint16_t src_port;
   uint16_t dst_port;
   uint16_t length;
   uint16_t checksum;
};

struct __attribute__((packed)) bf_comb_udp_hdr {
   bf_ethernet_hdr ethernet;
   bf_ipv4_hdr     ipv4;
   bf_udp_hdr      udp;
};


class VerbsSend {
    int         _fd;
    size_t      _pkt_size_max;
    int         _timeout;
    uint32_t    _rate_limit;
    bf_ibv_send _verbs;
    
    void get_interface_name(char* name) {
        sockaddr_in sin;
        socklen_t len = sizeof(sin);
        check_error(::getsockname(_fd, (sockaddr *)&sin, &len),
                    "query socket name");
        
        ifaddrs *ifaddr;
        check_error(::getifaddrs(&ifaddr), 
                    "query interfaces");
        for(ifaddrs *ifa=ifaddr; ifa != NULL; ifa=ifa->ifa_next) {
            if( ifa->ifa_addr != NULL) {
                if( reinterpret_cast<sockaddr_in*>(ifa->ifa_addr)->sin_addr.s_addr == sin.sin_addr.s_addr ) {
                    #pragma GCC diagnostic push
                    #pragma GCC diagnostic ignored "-Wstringop-truncation"
                    ::strncpy(name, ifa->ifa_name, IFNAMSIZ);
                    #pragma GCC diagnostic pop
                    break;
                }
            }
        }
        ::freeifaddrs(ifaddr);
    }
    void get_mac_address(uint8_t* mac) {
        ifreq ethreq;
        this->get_interface_name(&(ethreq.ifr_name[0]));
        check_error(::ioctl(_fd, SIOCGIFHWADDR, &ethreq),
                    "query interface hardware address");
        
        ::memcpy(mac, (uint8_t*) ethreq.ifr_hwaddr.sa_data, 6);
    }
    void get_remote_mac_address(uint8_t* mac) {
        uint32_t ip;
        char ip_str[INET_ADDRSTRLEN];
        this->get_remote_ip_address(&(ip_str[0]));
        ::inet_pton(AF_INET, &(ip_str[0]), &ip);
        
        if( ((ip & 0xFF) >= 224) && ((ip & 0xFF) < 240) ) {
            ETHER_MAP_IP_MULTICAST(&ip, mac);
        } else {
            int ret = -1;
            char cmd[256] = {'\0'};
            char line[256] = {'\0'};
            char ip_entry[INET_ADDRSTRLEN];
            unsigned int hw_type, flags;
            char mac_str[17];
            char mask[24];
            char dev[IFNAMSIZ];
            char* end;
            
            sprintf(cmd, "ping -c 1 %s", ip_str);
            FILE* fp = popen(cmd, "r");
            
            fp = fopen("/proc/net/arp", "r");
            while( ::fgets(line, sizeof(line), fp) != NULL) {
                ::sscanf(line, "%s 0x%x 0x%x %s %s %s\n",
                         ip_entry, &hw_type, &flags, mac_str, mask, dev);
                
                if( (::strcmp(ip_str, ip_entry) == 0) && (flags & ATF_COM) ) {
                    ret = 0;
                    mac[0] = (uint8_t) strtol(&mac_str[0], &end, 16);
                    mac[1] = (uint8_t) strtol(&mac_str[3], &end, 16);
                    mac[2] = (uint8_t) strtol(&mac_str[6], &end, 16);
                    mac[3] = (uint8_t) strtol(&mac_str[9], &end, 16);
                    mac[4] = (uint8_t) strtol(&mac_str[12], &end, 16);
                    mac[5] = (uint8_t) strtol(&mac_str[15], &end, 16);
                    break;
                }
            }
            fclose(fp);
            check_error(ret, "determine remote hardware address");    
        }
    }
    void get_ip_address(char* ip) {
        sockaddr_in sin;
        socklen_t len = sizeof(sin);
        check_error(::getsockname(_fd, (sockaddr *)&sin, &len),
                    "query socket name");
        inet_ntop(AF_INET, &(sin.sin_addr), ip, INET_ADDRSTRLEN);
    }
    void get_remote_ip_address(char* ip) {
        sockaddr_in sin;
        socklen_t len = sizeof(sin);
        check_error(::getpeername(_fd, (sockaddr *)&sin, &len),
                    "query peer name");
        inet_ntop(AF_INET, &(sin.sin_addr), ip, INET_ADDRSTRLEN);
    }
    uint16_t get_port() {
        sockaddr_in sin;
        socklen_t len = sizeof(sin);
        check_error(::getsockname(_fd, (sockaddr *)&sin, &len),
                    "query socket name");
        return ntohs(sin.sin_port);
    }
    uint16_t get_remote_port() {
        sockaddr_in sin;
        socklen_t len = sizeof(sin);
        check_error(::getpeername(_fd, (sockaddr *)&sin, &len),
                    "query peer name");
        return ntohs(sin.sin_port);
    }
    uint8_t get_ttl() {
      uint8_t ttl;
      socklen_t len = sizeof(ttl);
      check_error(::getsockopt(_fd, IPPROTO_IP, IP_TTL, &ttl, &len),
                  "determine TTL");
      return ttl;
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
    void create_context() {
        int d, p, g;
        int ndev, found;
        ibv_device** ibv_dev_list = NULL;
        ibv_context* ibv_ctx = NULL;
        ibv_device_attr_ex ibv_dev_attr;
        ibv_port_attr ibv_port_attr;
        union  ibv_gid ibv_gid;
        
        // Get the interface MAC address and GID
        found = 0;
        uint8_t mac[6] = {0};
        this->get_mac_address(&(mac[0]));
        uint64_t gid = this->get_interface_gid();
        
        // Find the right device
        /* Query all devices */
        ibv_dev_list = ibv_get_device_list(&ndev);
        check_null(ibv_dev_list,
                   "ibv_get_device_list");
        
        /* Interogate */
        for(d=0; d<ndev; d++) {
            ibv_ctx = ibv_open_device(ibv_dev_list[d]);
            check_null(ibv_ctx,
                       "open device");
            
            check_error(ibv_query_device_ex(ibv_ctx, NULL, &ibv_dev_attr),
                        "query device");
            
            /* Loop through the ports on the device */
            for(p=1; p<=ibv_dev_attr.orig_attr.phys_port_cnt; p++) {
                check_error(ibv_query_port(ibv_ctx, p, &ibv_port_attr),
                            "query port");
                
                /* Loop through GIDs of the port on the device */
                for(g=0; g<ibv_port_attr.gid_tbl_len; g++) {
                    check_error(ibv_query_gid(ibv_ctx, p, g, &ibv_gid),
                                "query gid on port");
                    
                    /* Did we find a match? */
                    if( (ibv_gid.global.subnet_prefix == 0x80feUL) \
                       && (ibv_gid.global.interface_id  == gid) ) {
                        found = 1;
                        
                        #if defined BF_ENABLE_VERBS_OFFLOAD && BF_ENABLE_VERBS_OFFLOAD
                        if( ibv_dev_attr.raw_packet_caps & IBV_RAW_PACKET_CAP_IP_CSUM ) {
                          _verbs.offload_csum = 1;
                        } else {
                          _verbs.offload_csum = 0;
                        }
                        #else
                        _verbs.offload_csum = 0;
                        #endif
                        std::cout << "_verbs.offload_csum: " << (int) _verbs.offload_csum << std::endl;
                        
                        _verbs.hardware_pacing[0] = _verbs.hardware_pacing[1] = 0;
                        #if defined BF_VERBS_SEND_PACING && BF_VERBS_SEND_PACING
                        if( ibv_is_qpt_supported(ibv_dev_attr.packet_pacing_caps.supported_qpts, IBV_QPT_RAW_PACKET) ) {
                          _verbs.hardware_pacing[0] = ibv_dev_attr.packet_pacing_caps.qp_rate_limit_min;  
                          _verbs.hardware_pacing[1] = ibv_dev_attr.packet_pacing_caps.qp_rate_limit_max;  
                        }
                        std::cout << "_verbs.hardware_pacing: " << (int) _verbs.hardware_pacing[0] << ", " << (int) _verbs.hardware_pacing[1] << std::endl;
                        #endif
                        break;
                    }
                }
                
                if( found ) {
                    break;
                }
            }
           
            if ( found ) {
                /* Save it to the class so that we can use it later */
                _verbs.ctx = ibv_ctx;
                _verbs.port_num = p;
                break;
            } else {
                check_error(ibv_close_device(ibv_ctx),
                            "close device");
            }
        }
        
        // Free the device list
        ibv_free_device_list(ibv_dev_list);
        
        // Done
        if( !found ) {
            destroy_context();
            throw VerbsSend::Error("specified device not found");
        }
    }
    void destroy_context() {
        int failures = 0;
        if( _verbs.ctx ) {
            if( ibv_close_device(_verbs.ctx) ) {
                failures += 1;
            }
        }
    }
    void create_buffers() {
        // Setup the protected domain
        _verbs.pd = ibv_alloc_pd(_verbs.ctx);
        
        // Create the buffers and the memory region
        _verbs.pkt_buf = (bf_ibv_send_pkt*) ::malloc(BF_VERBS_SEND_NPKTBUF*BF_VERBS_SEND_NQP * sizeof(struct bf_ibv_send_pkt));
        check_null(_verbs.pkt_buf, 
                   "allocate send packet buffer");
        ::memset(_verbs.pkt_buf, 0, BF_VERBS_SEND_NPKTBUF*BF_VERBS_SEND_NQP * sizeof(struct bf_ibv_send_pkt));
        _verbs.mr_size = (size_t) BF_VERBS_SEND_NPKTBUF*BF_VERBS_SEND_NQP * _pkt_size_max;
        _verbs.mr_buf = (uint8_t *) ::mmap(NULL, _verbs.mr_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_LOCKED, -1, 0);

        check_error(_verbs.mr_buf == MAP_FAILED,
                    "allocate send memory region buffer");
        check_error(::mlock(_verbs.mr_buf, _verbs.mr_size),
                    "lock send memory region buffer");
        _verbs.mr = ibv_reg_mr(_verbs.pd, _verbs.mr_buf, _verbs.mr_size, IBV_ACCESS_LOCAL_WRITE);
        check_null(_verbs.mr,
                   "register send memory region");
    }
    void destroy_buffers() {
        int failures = 0;
        if( _verbs.mr ) {
            if( ibv_dereg_mr(_verbs.mr) ) {
                failures += 1;
            }
        }
        
        if( _verbs.mr_buf ) {
            if( ::munlock(_verbs.mr_buf, _verbs.mr_size) ) {
                failures += 1;
            }
            if( ::munmap(_verbs.mr_buf, _verbs.mr_size) ) {
                failures += 1;
            }
        }
        
        if( _verbs.pkt_buf ) {
            free(_verbs.pkt_buf);
        }
        
        if( _verbs.pd ) {
            if( ibv_dealloc_pd(_verbs.pd) ) {
                failures += 1;
            }
        }
    }
    void create_queues() {
        int i;
        
        // Setup the completion channel and make it non-blocking
        _verbs.cc = ibv_create_comp_channel(_verbs.ctx);
        check_null(_verbs.cc,
                   "create send completion channel");
        int flags = ::fcntl(_verbs.cc->fd, F_GETFL);
        check_error(::fcntl(_verbs.cc->fd, F_SETFL, flags | O_NONBLOCK),
                    "set send completion channel to non-blocking");
        flags = ::fcntl(_verbs.cc->fd, F_GETFD);
        check_error(::fcntl(_verbs.cc->fd, F_SETFD, flags | O_CLOEXEC),
                    "set send completion channel to non-blocking");
        ::madvise(_verbs.cc, sizeof(ibv_pd), MADV_DONTFORK);
        
        // Setup the completion queues
        _verbs.cq = (ibv_cq**) ::malloc(BF_VERBS_SEND_NQP * sizeof(ibv_cq*));
        check_null(_verbs.cq,
                   "allocate send completion queues");
        ::memset(_verbs.cq, 0, BF_VERBS_SEND_NQP * sizeof(ibv_cq*));
        for(i=0; i<BF_VERBS_SEND_NQP; i++) {
            _verbs.cq[i] = ibv_create_cq(_verbs.ctx, BF_VERBS_SEND_NPKTBUF, NULL, NULL, 0);
            check_null(_verbs.cq[i],
                       "create send completion queue");
            
            // Request notifications before any receive completion can be created.
            // Do NOT restrict to solicited-only completions for receive.
            check_error(ibv_req_notify_cq(_verbs.cq[i], 0),
                        "change receive completion queue request notifications");
        }
        
        // Setup the queue pairs
        ibv_qp_init_attr qp_init;
        ::memset(&qp_init, 0, sizeof(ibv_qp_init_attr));
        qp_init.qp_context = NULL;
        qp_init.srq = NULL;
        qp_init.cap.max_send_wr = BF_VERBS_SEND_NPKTBUF;
        qp_init.cap.max_recv_wr = 0;
        qp_init.cap.max_send_sge = 1;
        qp_init.cap.max_recv_sge = 0;
        qp_init.cap.max_inline_data = 0;
        qp_init.qp_type = IBV_QPT_RAW_PACKET;
        qp_init.sq_sig_all = 1;
        
        _verbs.qp = (ibv_qp**) ::malloc(BF_VERBS_SEND_NQP*sizeof(ibv_qp*));
        check_null(_verbs.qp,
                   "allocate send queue pairs");
        ::memset(_verbs.qp, 0, BF_VERBS_SEND_NQP*sizeof(ibv_qp*));
        for(i=0; i<BF_VERBS_SEND_NQP; i++) {
            qp_init.send_cq = _verbs.cq[i];
            qp_init.recv_cq = _verbs.cq[i];
            _verbs.qp[i] = ibv_create_qp(_verbs.pd, &qp_init);
            check_null_qp(_verbs.qp[i],
                          "create send queue pair");
            
            // Transition queue pair to INIT state
            ibv_qp_attr qp_attr;
            ::memset(&qp_attr, 0, sizeof(ibv_qp_attr));
            qp_attr.qp_state = IBV_QPS_INIT;
            qp_attr.port_num = _verbs.port_num;
            
            check_error(ibv_modify_qp(_verbs.qp[i], &qp_attr, IBV_QP_STATE|IBV_QP_PORT),
                        "modify send queue pair state");
        }
    }
    void destroy_queues() {
        int failures = 0;
        
        if( _verbs.qp ) {
            for(int i=0; i<BF_VERBS_SEND_NQP; i++) {
                if( _verbs.qp[i] ) {
                    if( ibv_destroy_qp(_verbs.qp[i]) ) {
                        failures += 1;
                    }
                }
            }
            free(_verbs.qp);
        }
        
        if( _verbs.cc ) {
            if( ibv_destroy_comp_channel(_verbs.cc) ) {
                failures += 1;
            }
        }
        
        if( _verbs.cq ) {
            for(int i=0; i<BF_VERBS_SEND_NQP; i++) {
                if( _verbs.cq[i] ) {
                    if( ibv_destroy_cq(_verbs.cq[i]) ) {
                        failures += 1;
                    }
                }
            }
            free(_verbs.cq);
        }
    }
    void link_work_requests() {
        // Make sure we are ready to go
        check_null(_verbs.pkt_buf,
                   "find existing send packet buffer");
        check_null(_verbs.qp,
                   "find existing send queue pairs");
        
        // Setup the work requests
        int i, j;
        for(i=0; i<BF_VERBS_SEND_NPKTBUF*BF_VERBS_SEND_NQP; i++) {
            _verbs.pkt_buf[i].wr.wr_id = i;
            _verbs.pkt_buf[i].wr.num_sge = 1;
            _verbs.pkt_buf[i].sg.addr = (uint64_t) _verbs.mr_buf + i * _pkt_size_max;
            _verbs.pkt_buf[i].sg.length = _pkt_size_max;
            
            _verbs.pkt_buf[i].wr.sg_list = &(_verbs.pkt_buf[i].sg);
            for(j=0; j<_verbs.pkt_buf[i].wr.num_sge; j++) {
                _verbs.pkt_buf[i].wr.sg_list[j].lkey = _verbs.mr->lkey;
            }
        }
        
        // Link the work requests to send queue
        uint32_t send_flags = IBV_SEND_SIGNALED;
        #if defined BF_ENABLE_VERBS_OFFLOAD && BF_ENABLE_VERBS_OFFLOAD
        if( _verbs.offload_csum ) {
          send_flags |= IBV_SEND_IP_CSUM;
        }
        #endif
        
        for(i=0; i<BF_VERBS_SEND_NQP*BF_VERBS_SEND_NPKTBUF-1; i++) {
            _verbs.pkt_buf[i].wr.next = &(_verbs.pkt_buf[i+1].wr);
            _verbs.pkt_buf[i].wr.opcode = IBV_WR_SEND;
            _verbs.pkt_buf[i].wr.send_flags = send_flags;
        }
        _verbs.pkt_buf[BF_VERBS_SEND_NQP*BF_VERBS_SEND_NPKTBUF-1].wr.next = NULL;
        _verbs.pkt_buf[BF_VERBS_SEND_NQP*BF_VERBS_SEND_NPKTBUF-1].wr.opcode = IBV_WR_SEND;
        _verbs.pkt_buf[BF_VERBS_SEND_NQP*BF_VERBS_SEND_NPKTBUF-1].wr.send_flags = send_flags;
        _verbs.pkt_head = _verbs.pkt_buf;
    }
    inline bf_ibv_send_pkt* get_packet_buffers(int npackets) {
        int i, j;
        int num_wce;
        ibv_qp_attr qp_attr;
        ibv_cq *ev_cq;
        intptr_t ev_cq_ctx;
        ibv_wc wc[BF_VERBS_SEND_WCBATCH];
        bf_ibv_send_pkt *send_pkt = NULL;
        bf_ibv_send_pkt *send_head = NULL;
        bf_ibv_send_pkt *send_tail = NULL;
        
        // Ensure the queue pairs are in a state suitable for receiving
        for(i=0; i<BF_VERBS_SEND_NQP; i++) {
            switch(_verbs.qp[i]->state) {
                case IBV_QPS_RESET: // Unexpected, but maybe user reset it
                    qp_attr.qp_state = IBV_QPS_INIT;
                    qp_attr.port_num = _verbs.port_num;
                    if( ibv_modify_qp(_verbs.qp[i], &qp_attr, IBV_QP_STATE|IBV_QP_PORT) ) {
                        return NULL;
                    }
                case IBV_QPS_INIT:
                    qp_attr.qp_state = IBV_QPS_RTR;
                    qp_attr.port_num = _verbs.port_num;
                    if( ibv_modify_qp(_verbs.qp[i], &qp_attr, IBV_QP_STATE) ) {
                        return NULL;
                    }
                case IBV_QPS_RTR:
                    qp_attr.qp_state = IBV_QPS_RTS;
                    if(ibv_modify_qp(_verbs.qp[i], &qp_attr, IBV_QP_STATE)) {
                        return NULL;
                    }
                    break;
                case IBV_QPS_RTS:
                    break;
                default:
                    return NULL;
            }
        }
        
        // Get the completion event(s)
        while( ibv_get_cq_event(_verbs.cc, &ev_cq, (void **)&ev_cq_ctx) == 0 ) {
            // Ack the event
            ibv_ack_cq_events(ev_cq, 1);
        }
        
        for(i=0; i<BF_VERBS_SEND_NQP; i++) {
          // Request notification upon the next completion event
          // Do NOT restrict to solicited-only completions
          if( ibv_req_notify_cq(_verbs.cq[i], 0) ) {
              return NULL;
           }
        }
        
        while(_verbs.nqueued > 0 ) {
            for(i=0; i<BF_VERBS_SEND_NQP; i++) {
                do {
                    num_wce = ibv_poll_cq(_verbs.cq[i], BF_VERBS_SEND_WCBATCH, &wc[0]);
                    if(num_wce < 0) {
                        return NULL;
                    }
                    
                    // Loop through all work completions
                    for(j=0; j<num_wce; j++) {
                        send_pkt = &(_verbs.pkt_buf[wc[j].wr_id]);
                        send_pkt->wr.next = &(_verbs.pkt_head->wr);
                        _verbs.pkt_head = send_pkt;
                    } // for each work completion
                    
                    // Decrement the number of packet buffers in use
                    _verbs.nqueued -= num_wce;
                } while(num_wce);
            }
        }
        
        if( npackets == 0 || !_verbs.pkt_head ) {
            return NULL;
        }
        
        send_head = _verbs.pkt_head;
        send_tail = _verbs.pkt_head;
        for(i=0; i<npackets-1 && send_tail->wr.next; i++) {
          send_tail = (bf_ibv_send_pkt*) send_tail->wr.next;
        }
        
        _verbs.pkt_head = (bf_ibv_send_pkt*) send_tail->wr.next;
        send_tail->wr.next = NULL;
        
        _verbs.nqueued += npackets;
        
        return send_head;
    }
    inline void check_error(int retval, std::string what) {
        if( retval < 0 ) {
            destroy_queues();
            destroy_buffers();
            destroy_context();
            
            std::stringstream ss;
            ss << "Failed to " << what << ": (" << errno << ") "
               << strerror(errno);
            throw VerbsSend::Error(ss.str());
        }
    }
    inline void check_null(void* ptr, std::string what) {
        if( ptr == NULL ) {
            destroy_queues();
            destroy_buffers();
            destroy_context();
            
            std::stringstream ss;
            ss << "Failed to " << what << ": (" << errno << ") "
               << strerror(errno);
            throw VerbsSend::Error(ss.str());
        }
    }
    inline void check_null_qp(void* ptr, std::string what) {
        if( ptr == NULL ) {
            destroy_queues();
            destroy_buffers();
            destroy_context();
            
            std::stringstream ss;
            ss << "Failed to " << what << ": (" << errno << ") "
               << strerror(errno);
            if( errno == EPERM ) {
              ss << "  Do you need to set 'options ibverbs disable_raw_qp_enforcement=1' " 
                 << "or add the CAP_NET_RAW capability?";
            }
            throw VerbsSend::Error(ss.str());
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
    
    VerbsSend(int fd, size_t pkt_size_max)
        : _fd(fd), _pkt_size_max(pkt_size_max), _timeout(1), _rate_limit(0) {
            _timeout = get_timeout_ms();
            
            ::memset(&_verbs, 0, sizeof(_verbs));
            check_error(ibv_fork_init(),
                        "make verbs fork safe");
            create_context();
            create_buffers();
            create_queues();
            link_work_requests();
    }
    ~VerbsSend() {
        destroy_queues();
        destroy_buffers();
        destroy_context();
    }
    inline void set_rate_limit(uint32_t rate_limit, size_t udp_length, size_t max_burst_size=BF_VERBS_SEND_NPKTBURST) {
      int i;
      
      // Converts to B/s to kb/s assuming a packet size
      size_t pkt_size = udp_length + BF_VERBS_SEND_PAYLOAD_OFFSET;
      rate_limit = ((float) rate_limit) / udp_length * pkt_size * 8 / 1000;
      
      // Verify that this rate limit is valid
      if( rate_limit == 0 ) {
        rate_limit = _verbs.hardware_pacing[1];
      }
      if( rate_limit < _verbs.hardware_pacing[0] || rate_limit > _verbs.hardware_pacing[1] ) {
        throw VerbsSend::Error("Failed to set rate limit, specified rate limit is out of range");
      }
      
      // Apply the rate limit
      #if defined BF_VERBS_SEND_PACING && BF_VERBS_SEND_PACING
      ibv_qp_rate_limit_attr rl_attr;
      ::memset(&rl_attr, 0, sizeof(ibv_qp_rate_limit_attr));
      rl_attr.rate_limit = rate_limit;
      rl_attr.typical_pkt_sz = pkt_size;
      rl_attr.max_burst_sz = max_burst_size*pkt_size;
      for(i=0; i<BF_VERBS_SEND_NQP; i++) {
          check_error(ibv_modify_qp_rate_limit(_verbs.qp[i], &rl_attr),
                      "set queue pair rate limit");
      }
      #endif
      
      _rate_limit = rate_limit;
    }
    inline void get_ethernet_header(bf_ethernet_hdr* hdr) {
        uint8_t src_mac[6], dst_mac[6];
        this->get_mac_address(&(src_mac[0]));
        this->get_remote_mac_address(&(dst_mac[0]));
        
        ::memset(hdr, 0, sizeof(bf_ethernet_hdr));
        ::memcpy(hdr->dst_mac, dst_mac, 6);
        ::memcpy(hdr->src_mac, src_mac, 6);
        hdr->type = htons(0x0800);  // IPv4
    }
    inline void get_ipv4_header(bf_ipv4_hdr* hdr, size_t udp_length=0) {
        uint8_t ttl = this->get_ttl();
        uint32_t src_ip, dst_ip;
        char src_ip_str[INET_ADDRSTRLEN], dst_ip_str[INET_ADDRSTRLEN];
        this->get_ip_address(&(src_ip_str[0]));
        inet_pton(AF_INET, &(src_ip_str[0]), &src_ip);
        this->get_remote_ip_address(&(dst_ip_str[0]));
        inet_pton(AF_INET, &(dst_ip_str[0]), &dst_ip);
        
        ::memset(hdr, 0, sizeof(bf_ipv4_hdr));
        hdr->version_ihl = htons(0x4500);   // v4 + 20-byte header
        hdr->length = htons((uint16_t) (20 + 8 + udp_length));
        hdr->flags_frag = htons(1<<14);     // don't fragment
        hdr->ttl = ttl;
        hdr->proto = 0x11;                  // UDP
        hdr->src_addr = src_ip;
        hdr->dst_addr = dst_ip;
        if( !_verbs.offload_csum ) {
          bf_ipv4_update_checksum(hdr);
        }
    }
    inline void get_udp_header(bf_udp_hdr* hdr, size_t udp_length=0) {
       uint16_t src_port, dst_port;
       src_port = this->get_port();
       dst_port = this->get_remote_port();
       
       ::memset(hdr, 0, sizeof(bf_udp_hdr));
       hdr->src_port = htons(src_port);
       hdr->dst_port = htons(dst_port);
       hdr->length = htons((uint16_t) (8 + udp_length));
    }
    inline int sendmmsg(mmsghdr *mmsg, int npackets, int flags=0) {
      int ret;
      bf_ibv_send_pkt* head;
      ibv_send_wr *s;
      
      if( npackets > BF_VERBS_SEND_NPKTBUF ) {
        throw VerbsSend::Error(std::string("Too many packets for the current buffer size"));
      }
      
      // Reclaim a set of buffers to use
      head = this->get_packet_buffers(npackets);
      
      // Load in the new data
      int i;
      uint64_t offset;
      for(i=0; i<npackets; i++) {
          offset = 0;
          ::memcpy(_verbs.mr_buf + i * _pkt_size_max + offset,
                   mmsg[i].msg_hdr.msg_iov[0].iov_base,
                   mmsg[i].msg_hdr.msg_iov[0].iov_len);
          offset += mmsg[i].msg_hdr.msg_iov[0].iov_len;
          ::memcpy(_verbs.mr_buf + i * _pkt_size_max + offset,
                   mmsg[i].msg_hdr.msg_iov[1].iov_base,
                   mmsg[i].msg_hdr.msg_iov[1].iov_len);
          offset += mmsg[i].msg_hdr.msg_iov[1].iov_len;
          ::memcpy(_verbs.mr_buf + i * _pkt_size_max + offset,
                   mmsg[i].msg_hdr.msg_iov[2].iov_base,
                   mmsg[i].msg_hdr.msg_iov[2].iov_len);
          offset += mmsg[i].msg_hdr.msg_iov[2].iov_len;
          _verbs.pkt_buf[i].sg.length = offset;
      }
      
      // Queue for sending
      ret = ibv_post_send(_verbs.qp[0], &(head->wr), &s);
      if( ret ) {
        ret = -1;
      } else {
        ret = npackets;
      }
      return ret;
    }
};
