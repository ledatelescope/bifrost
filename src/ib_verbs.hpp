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
#include <netinet/if_ether.h>
#include <sys/mman.h>
#include <errno.h>

#include <infiniband/verbs.h>

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

struct bf_ibv_recv_pkt{
    ibv_recv_wr wr;
    ibv_sge     sg;
    uint32_t    length;
};

struct bf_ibv_flow {
    ibv_flow_attr         attr;
    ibv_flow_spec_eth     eth;
    ibv_flow_spec_ipv4    ipv4;
    ibv_flow_spec_tcp_udp udp;
} __attribute__((packed));

struct bf_ibv {
    ibv_context*      ctx;
    uint8_t           port_num;
    ibv_pd*           pd;
    ibv_comp_channel* cc; 
    ibv_cq**          cq;
    ibv_qp**          qp;
    ibv_flow**        flows;
    
    uint8_t*          mr_buf;
    size_t            mr_size;
    ibv_mr*           mr;
    
    bf_ibv_recv_pkt*  pkt_buf;
    bf_ibv_recv_pkt*  pkt;
    bf_ibv_recv_pkt*  pkt_batch;
};


class Verbs {
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
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wstringop-truncation"
            strncpy(name, &(line[0]), IFNAMSIZ);
            #pragma GCC diagnostic pop
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
                #pragma GCC diagnostic push
                #pragma GCC diagnostic ignored "-Wstringop-truncation"
                strncpy(name, &(line[0]), IFNAMSIZ);
                #pragma GCC diagnostic pop
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
            throw Verbs::Error("specified device not found");
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
        _verbs.pkt_buf = (bf_ibv_recv_pkt*) ::malloc(BF_VERBS_NPKTBUF*BF_VERBS_NQP * sizeof(struct bf_ibv_recv_pkt));
        check_null(_verbs.pkt_buf, 
                   "allocate receive packet buffer");
        ::memset(_verbs.pkt_buf, 0, BF_VERBS_NPKTBUF*BF_VERBS_NQP * sizeof(struct bf_ibv_recv_pkt));
        _verbs.mr_size = (size_t) BF_VERBS_NPKTBUF*BF_VERBS_NQP * _pkt_size_max;
        _verbs.mr_buf = (uint8_t *) ::mmap(NULL, _verbs.mr_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_LOCKED, -1, 0);

        check_error(_verbs.mr_buf == MAP_FAILED,
                    "allocate receive memory region buffer");
        check_error(::mlock(_verbs.mr_buf, _verbs.mr_size),
                    "lock receive memory region buffer");
        _verbs.mr = ibv_reg_mr(_verbs.pd, _verbs.mr_buf, _verbs.mr_size, IBV_ACCESS_LOCAL_WRITE);
        check_null(_verbs.mr,
                   "register receive memory region");
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
                   "create receive completion channel");
        int flags = ::fcntl(_verbs.cc->fd, F_GETFL);
        check_error(::fcntl(_verbs.cc->fd, F_SETFL, flags | O_NONBLOCK),
                    "set receive completion channel to non-blocking");
        flags = ::fcntl(_verbs.cc->fd, F_GETFD);
        check_error(::fcntl(_verbs.cc->fd, F_SETFD, flags | O_CLOEXEC),
                    "set receive completion channel to non-blocking");
        ::madvise(_verbs.cc, sizeof(ibv_pd), MADV_DONTFORK);
        
        // Setup the completion queues
        _verbs.cq = (ibv_cq**) ::malloc(BF_VERBS_NQP * sizeof(ibv_cq*));
        check_null(_verbs.cq,
                   "allocate receive completion queues");
        ::memset(_verbs.cq, 0, BF_VERBS_NQP * sizeof(ibv_cq*));
        for(i=0; i<BF_VERBS_NQP; i++) {
            _verbs.cq[i] = ibv_create_cq(_verbs.ctx, BF_VERBS_NPKTBUF, NULL, _verbs.cc, 0);
            check_null(_verbs.cq[i],
                       "create receive completion queue");
            
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
        qp_init.cap.max_send_wr = 0;
        qp_init.cap.max_recv_wr = BF_VERBS_NPKTBUF;
        qp_init.cap.max_send_sge = 0;
        qp_init.cap.max_recv_sge = 1;
        qp_init.cap.max_inline_data = 0;
        qp_init.qp_type = IBV_QPT_RAW_PACKET;
        qp_init.sq_sig_all = 0;
        
        _verbs.qp = (ibv_qp**) ::malloc(BF_VERBS_NQP*sizeof(ibv_qp*));
        check_null(_verbs.qp,
                   "allocate receive queue pairs");
        ::memset(_verbs.qp, 0, BF_VERBS_NQP*sizeof(ibv_qp*));
        for(i=0; i<BF_VERBS_NQP; i++) {
            qp_init.send_cq = _verbs.cq[i];
            qp_init.recv_cq = _verbs.cq[i];
            _verbs.qp[i] = ibv_create_qp(_verbs.pd, &qp_init);
            check_null_qp(_verbs.qp[i],
                          "create receive queue pair");
            
            // Transition queue pair to INIT state
            ibv_qp_attr qp_attr;
            ::memset(&qp_attr, 0, sizeof(ibv_qp_attr));
            qp_attr.qp_state = IBV_QPS_INIT;
            qp_attr.port_num = _verbs.port_num;
            
            check_error(ibv_modify_qp(_verbs.qp[i], &qp_attr, IBV_QP_STATE|IBV_QP_PORT),
                        "modify receive queue pair state");
        }
    }
    void destroy_queues() {
        int failures = 0;
        
        if( _verbs.qp ) {
            for(int i=0; i<BF_VERBS_NQP; i++) {
                if( _verbs.qp[i] ) {
                    if( ibv_destroy_qp(_verbs.qp[i]) ) {
                        failures += 1;
                    }
                }
            }
            free(_verbs.qp);
        }
        
        if( _verbs.cq ) {
            for(int i=0; i<BF_VERBS_NQP; i++) {
                if( _verbs.cq[i] ) {
                    if( ibv_destroy_cq(_verbs.cq[i]) ) {
                        failures += 1;
                    }
                }
            }
            free(_verbs.cq);
        }
        
        if( _verbs.cc ) {
            if( ibv_destroy_comp_channel(_verbs.cc) ) {
                failures += 1;
            }
        }
    }
    void link_work_requests() {
        // Make sure we are ready to go
        check_null(_verbs.pkt_buf,
                   "find existing receive packet buffer");
        check_null(_verbs.qp,
                   "find existing receive queue pairs");
        
        // Setup the work requests
        int i, j, k;
        ibv_recv_wr* recv_wr_bad;
        
        for(i=0; i<BF_VERBS_NPKTBUF*BF_VERBS_NQP; i++) {
            _verbs.pkt_buf[i].wr.wr_id = i;
            _verbs.pkt_buf[i].wr.num_sge = 1;
            _verbs.pkt_buf[i].sg.addr = (uint64_t) _verbs.mr_buf + i * _pkt_size_max;
            _verbs.pkt_buf[i].sg.length = _pkt_size_max;
            
            _verbs.pkt_buf[i].wr.sg_list = &(_verbs.pkt_buf[i].sg);
            for(j=0; j<_verbs.pkt_buf[i].wr.num_sge; j++) {
                _verbs.pkt_buf[i].wr.sg_list[j].lkey = _verbs.mr->lkey;
            }
        }
        
        // Link the work requests to receive queue
        for(i=0; i<BF_VERBS_NQP; i++) {
            k = i*BF_VERBS_NPKTBUF;
            for(j=0; j<BF_VERBS_NPKTBUF-1; j++) {
                _verbs.pkt_buf[k+j].wr.next = &(_verbs.pkt_buf[k+j+1].wr);
            }
            _verbs.pkt_buf[k+BF_VERBS_NPKTBUF-1].wr.next = NULL;
            
            // Post work requests to the receive queue
            check_error(ibv_post_recv(_verbs.qp[i], &_verbs.pkt_buf[k].wr, &recv_wr_bad),
                        "post work request to receive queue");
        }
    }
    void create_flows() {
        // Setup the flows
        int i;
        
        bf_ibv_flow flow;
        ::memset(&flow, 0, sizeof(flow));
        
        flow.attr.comp_mask = 0;
        flow.attr.type = IBV_FLOW_ATTR_NORMAL;
        flow.attr.size = sizeof(flow);
        flow.attr.priority = 0;
        flow.attr.num_of_specs = 3;
        flow.attr.port = _verbs.port_num;
        flow.attr.flags = 0;
        flow.eth.type = IBV_FLOW_SPEC_ETH;
        flow.eth.size = sizeof(flow.eth);
        flow.ipv4.type = IBV_FLOW_SPEC_IPV4;
        flow.ipv4.size = sizeof(flow.ipv4);
        flow.udp.type = IBV_FLOW_SPEC_UDP;
        flow.udp.size = sizeof(flow.udp);

        // Filter on UDP and the port
        flow.udp.val.dst_port = htons(this->get_port());
        flow.udp.mask.dst_port = 0xffff;
        
        // Filter on IP address in the IPv4 header
        uint32_t ip;
        char ip_str[INET_ADDRSTRLEN];
        this->get_ip_address(&(ip_str[0]));
        inet_pton(AF_INET, &(ip_str[0]), &ip);
        ::memcpy(&(flow.ipv4.val.dst_ip), &ip, 4);
        ::memset(&(flow.ipv4.mask.dst_ip), 0xff, 4);
        
        // Filter on the destination MAC address (actual or multicast)
        uint8_t mac[6] = {0};
        this->get_mac_address(&(mac[0]));
        if( ((ip & 0xFF) >= 224) && ((ip & 0xFF) < 240) ) {
            ETHER_MAP_IP_MULTICAST(&ip, mac);
        }
        ::memcpy(&flow.eth.val.dst_mac, &mac, 6);
        ::memset(&flow.eth.mask.dst_mac, 0xff, 6);
        
        // Create the flows
        _verbs.flows = (ibv_flow**) ::malloc(BF_VERBS_NQP * sizeof(ibv_flow*));
        check_null(_verbs.flows,
                   "allocate flows");
        ::memset(_verbs.flows, 0, BF_VERBS_NQP * sizeof(ibv_flow*));
        for(i=0; i<BF_VERBS_NQP; i++) {
            
            _verbs.flows[i] = ibv_create_flow(_verbs.qp[i], &flow.attr);
            check_null(_verbs.flows[i],
                       "create flow");
        }
    }
    void destroy_flows() {
        int failures = 0;
        
        if( _verbs.flows ) {
            for(int i=0; i<BF_VERBS_NQP; i++) {
                if( _verbs.flows[i] ) {
                    if( ibv_destroy_flow(_verbs.flows[i]) ) {
                        failures += 1;
                    }
                }
            }
            free(_verbs.flows);
        }
    }
    inline int release(bf_ibv_recv_pkt* recv_pkt) {
        int i;
        ibv_recv_wr* recv_wr_bad;
        
        if( !recv_pkt ) {
            return 0;
        }
        
        // Figure out which QP these packets belong to and repost to that QP
        i = recv_pkt->wr.wr_id / BF_VERBS_NPKTBUF;
        return ibv_post_recv(_verbs.qp[i], &recv_pkt->wr, &recv_wr_bad);
    }
    inline bf_ibv_recv_pkt* receive() {
        int i;
        int num_wce;
        uint64_t wr_id;
        pollfd pfd;
        ibv_qp_attr qp_attr;
        ibv_cq *ev_cq;
        intptr_t ev_cq_ctx;
        ibv_wc wc[BF_VERBS_WCBATCH];
        bf_ibv_recv_pkt *recv_head = NULL;
        ibv_recv_wr *recv_tail = NULL;
        
        // Ensure the queue pairs are in a state suitable for receiving
        for(i=0; i<BF_VERBS_NQP; i++) {
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
                    break;
                case IBV_QPS_RTR:
                case IBV_QPS_RTS:
                    break;
                default:
                    return NULL;
            }
        }
        
        // Setup for poll
        pfd.fd = _verbs.cc->fd;
        pfd.events = POLLIN;
        pfd.revents = 0;
        
        // poll completion channel's fd with given timeout
        int rc = ::poll(&pfd, 1, _timeout);
        if( rc == 0) {
            // Timeout
            errno = EAGAIN;
            throw Verbs::Error("timeout");
        } else if( rc < 0) {
            // Error
            throw Verbs::Error("error");
        }
        
        // Get the completion event
        if( ibv_get_cq_event(_verbs.cc, &ev_cq, (void **)&ev_cq_ctx) ) {
            return NULL;
        }
        
        // Ack the event
        ibv_ack_cq_events(ev_cq, 1);
        
        // Request notification upon the next completion event
        // Do NOT restrict to solicited-only completions
        if( ibv_req_notify_cq(ev_cq, 0) ) {
            return NULL;
        }
        
        // Empty the CQ: poll all of the completions from the CQ (if any exist)
        do {
            num_wce = ibv_poll_cq(ev_cq, BF_VERBS_WCBATCH, &wc[0]);
            if( num_wce < 0 ) {
                return NULL;
            }
            
            // Loop through all work completions
            for(i=0; i<num_wce; i++) {
                wr_id = wc[i].wr_id;
                // Set length to BF_VERBS_PAYLOAD_OFFSET for unsuccessful work requests
                if( wc[i].status != IBV_WC_SUCCESS ) {
                    _verbs.pkt_buf[wr_id].length = BF_VERBS_PAYLOAD_OFFSET;
                } else {
                    // Copy byte_len from completion to length of pkt srtuct
                    _verbs.pkt_buf[wr_id].length = wc[i].byte_len;
                }
                // Add work requests to recv list
                if( !recv_head ) {
                    recv_head = &(_verbs.pkt_buf[wr_id]);
                    recv_tail = &recv_head->wr;
                } else {
                    recv_tail->next = &(_verbs.pkt_buf[wr_id].wr);
                    recv_tail = recv_tail->next;
                }
            } // for each work completion
        } while(num_wce);
        
        // Ensure list is NULL terminated (if we have a list)
        if(recv_tail) {
            recv_tail->next = NULL;
        }
        
        return recv_head;
    }
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
    inline void check_null_qp(void* ptr, std::string what) {
        if( ptr == NULL ) {
            destroy_flows();
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
        : _fd(fd), _pkt_size_max(pkt_size_max), _timeout(1) {
            _timeout = get_timeout_ms();
            
            ::memset(&_verbs, 0, sizeof(_verbs));
            check_error(ibv_fork_init(),
                        "make verbs fork safe");
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
    inline int recv_packet(uint8_t** pkt_ptr, int flags=0) {
      // If we don't have a work-request queue on the go,
      // get some new packets.
      if( _verbs.pkt != NULL) {
          _verbs.pkt = (bf_ibv_recv_pkt *) _verbs.pkt->wr.next;
          if( _verbs.pkt == NULL ) {
              this->release(_verbs.pkt_batch);
              _verbs.pkt_batch = NULL;
          }
      }
      
      while( _verbs.pkt_batch == NULL ) {
          try {
              _verbs.pkt_batch = this->receive();
          } catch(Verbs::Error const&) {
              _verbs.pkt = NULL;
              errno = EAGAIN;
              return -1;
          }
          _verbs.pkt = _verbs.pkt_batch;
      }
      
      // IBV returns Eth/UDP/IP headers. Strip them off here
      *pkt_ptr = (uint8_t *) _verbs.pkt->wr.sg_list->addr + BF_VERBS_PAYLOAD_OFFSET;
      return _verbs.pkt->length - BF_VERBS_PAYLOAD_OFFSET;
    }
};
