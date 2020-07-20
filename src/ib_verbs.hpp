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
#define BF_VERBS_NPKTBUF 4096
#endif

#ifndef BF_VERBS_WCBATCH
#define BF_VERBS_WCBATCH 16
#endif

#define BF_VERBS_PAYLOAD_OFFSET 42

struct bf_ibv_recv_pkt {
    struct ibv_recv_wr wr;
    uint32_t           length;
};

// Structure for defining various types of flow rules
struct bf_ibv_flow {
    struct ibv_flow_attr         attr;
    struct ibv_flow_spec_tcp_udp spec_tcp_udp;
    struct ibv_flow_spec_ipv4    spec_ipv4;
    struct ibv_flow_spec_eth     spec_eth;
} __attribute__((packed));

class Verbs {
    int                      _fd;
    size_t                   _pkt_size_max;
    
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
    inline int recv_packet(uint8_t** pkt_ptr, int flags=0) {
        // If we don't have a work-request queue on the go,
        // get some new packets.
        if( _pkt ) {
            _pkt = (struct bf_ibv_recv_pkt *)_pkt->wr.next;
            if( !_pkt ) {
                this->release(_chain);
                _chain = NULL;
            }
        }
        while( !_chain ) {
            _chain = this->receive(500);
            _pkt = _chain;
        }
        // IBV returns Eth/UDP/IP headers. Strip them off here.
        *pkt_ptr = (uint8_t *)_pkt->wr.sg_list->addr + BF_VERBS_PAYLOAD_OFFSET;
        return _pkt->length - BF_VERBS_PAYLOAD_OFFSET;
    }
};

void Verbs::create_context() {
    int d, p, g;
    int ndev, found;
    struct ibv_device** ibv_dev_list = NULL;
    struct ibv_context* ibv_ctx = NULL;
    struct ibv_device_attr ibv_dev_attr;
    struct ibv_port_attr ibv_port_attr;
    union  ibv_gid ibv_gid;
    
    // Get the interface MAC address and GID
    found = 0;
    uint8_t mac[6] = {0};
    this->get_mac_address(&(mac[0]));
    uint64_t gid = this->get_interface_gid();
    
    std::cout << "MAC: " << std::hex << int(mac[0]) << ":" << int(mac[1]) << ":" << int(mac[2]) 
                              << ":" << int(mac[3]) << ":" << int(mac[4]) << ":" << int(mac[5]) << std::dec << std::endl;
    std::cout << "GID: " << std::hex << int(htons( gid        & 0xFFFF)) << ":" << int(htons((gid >> 16) & 0xFFFF))
                              << ":" << int(htons((gid >> 32) & 0xFFFF)) << ":" << int(htons((gid >> 48) & 0xFFFF)) << std::dec << std::endl;
    
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
        
        check_error(ibv_query_device(ibv_ctx, &ibv_dev_attr),
                    "query device");
        
        /* Loop through the ports on the device */
        for(p=1; p<=ibv_dev_attr.phys_port_cnt; p++) {
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
            _ctx = ibv_ctx;
            ::memcpy(&_dev_attr, &ibv_dev_attr, sizeof(struct ibv_device_attr));
            _port_num = p;
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

void Verbs::destroy_context() {
    int failures = 0;
    if( _ctx ) {
        if( ibv_close_device(_ctx) ) {
            failures += 1;
        }
    }
}

void Verbs::create_buffers() {
    // Setup the protected domain
    _pd = ibv_alloc_pd(_ctx);
    
    // Create the buffers, the scatter/gather entries, and the memory region
    _pkt_buf = (struct bf_ibv_recv_pkt*) ::malloc(BF_VERBS_NPKTBUF*BF_VERBS_NQP * sizeof(struct bf_ibv_recv_pkt));
    check_null(_pkt_buf, 
               "allocate receive packet buffer");
    ::memset(_pkt_buf, 0, BF_VERBS_NPKTBUF*BF_VERBS_NQP * sizeof(bf_ibv_recv_pkt));
    _sge = (struct ibv_sge*) ::malloc(BF_VERBS_NPKTBUF*BF_VERBS_NQP * sizeof(struct ibv_sge));
    check_null(_sge,
               "allocate scatter/gather entries");
    ::memset(_sge, 0, BF_VERBS_NPKTBUF*BF_VERBS_NQP * sizeof(struct ibv_sge));
    _mr_size = (size_t) BF_VERBS_NPKTBUF*BF_VERBS_NQP * _pkt_size_max;
    _mr_buf = (uint8_t *) ::malloc(_mr_size);
    check_null(_mr_buf,
               "allocate memory region buffer");
    ::memset(_mr_buf, 0, _mr_size);
    _mr = ibv_reg_mr(_pd, _mr_buf, _mr_size, IBV_ACCESS_LOCAL_WRITE);
    check_null(_mr,
               "register memory region");
}

void Verbs::destroy_buffers() {
    int failures = 0;
    if( _mr ) {
        if( ibv_dereg_mr(_mr) ) {
            failures += 1;
        }
    }
    
    if( _mr_buf ) {
        free(_mr_buf);
    }
    
    if( _sge ) {
        free(_sge);
    }
    
    if( _pkt_buf ) {
        free(_pkt_buf);
    }
    
    if( _pd ) {
        if( ibv_dealloc_pd(_pd) ) {
            failures += 1;
        }
    }
}

void Verbs::create_queues() {
    int i;
    
    // Setup the completion channel and make it non-blocking
    _cc = ibv_create_comp_channel(_ctx);
    check_null(_cc,
               "create completion channel");
    int flags = ::fcntl(_cc->fd, F_GETFL);
    check_error(::fcntl(_cc->fd, F_SETFL, flags | O_NONBLOCK),
                "set completion channel to non-blocking");
    
    // Setup the completion queues
    _cq = (struct ibv_cq**) ::malloc(BF_VERBS_NQP * sizeof(struct ibv_cq*));
    check_null(_cq,
               "allocate completion queues");
    ::memset(_cq, 0, BF_VERBS_NQP * sizeof(struct ibv_cq*));
    for(i=0; i<BF_VERBS_NQP; i++) {
        _cq[i] = ibv_create_cq(_ctx, BF_VERBS_NPKTBUF, (void*)(uintptr_t) i, _cc, 0);
        check_null(_cq[i],
                   "create completion queue");
        
        // Request notifications before any receive completion can be created.
        // Do NOT restrict to solicited-only completions for receive.
        check_error(ibv_req_notify_cq(_cq[i], 0),
                    "change completion queue request notifications");
    }
    
    // Setup the queue pairs
    struct ibv_qp_init_attr qp_init;
    ::memset(&qp_init, 0, sizeof(struct ibv_qp_init_attr));
    qp_init.qp_context = NULL;
    qp_init.srq = NULL;
    qp_init.cap.max_send_wr = 0;
    qp_init.cap.max_recv_wr = BF_VERBS_NPKTBUF;
    qp_init.cap.max_send_sge = 0;
    qp_init.cap.max_recv_sge = 1;
    qp_init.cap.max_inline_data = 0;
    qp_init.qp_type = IBV_QPT_RAW_PACKET;
    qp_init.sq_sig_all = 0;
    
    _qp = (struct ibv_qp**) ::malloc(BF_VERBS_NQP*sizeof(struct ibv_qp*));
    check_null(_qp,
               "allocate queue pairs");
    ::memset(_qp, 0, BF_VERBS_NQP*sizeof(struct ibv_qp*));
    for(i=0; i<BF_VERBS_NQP; i++) {
        qp_init.send_cq = _cq[i];
        qp_init.recv_cq = _cq[i];
        _qp[i] = ibv_create_qp(_pd, &qp_init);
        check_null(_qp[i],
                   "create queue pair");
        
        // Transition queue pair to INIT state
        struct ibv_qp_attr qp_attr;
        ::memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
        qp_attr.qp_state = IBV_QPS_INIT;
        qp_attr.port_num = _port_num;
        
        check_error(ibv_modify_qp(_qp[i], &qp_attr, IBV_QP_STATE|IBV_QP_PORT),
                    "modify queue pair state");
    }

}

void Verbs::destroy_queues() {
    int failures = 0;
    
    if( _qp ) {
        for(int i=0; i<BF_VERBS_NQP; i++) {
            if( _qp[i] ) {
                if( ibv_destroy_qp(_qp[i]) ) {
                    failures += 1;
                }
            }
        }
        free(_qp);
    }
    
    if( _cq ) {
        for(int i=0; i<BF_VERBS_NQP; i++) {
            if( _cq[i] ) {
                if( ibv_destroy_cq(_cq[i]) ) {
                    failures += 1;
                }
            }
        }
        free(_cq);
    }
    
    if( _cc ) {
        if( ibv_destroy_comp_channel(_cc) ) {
            failures += 1;
        }
    }
}

void Verbs::link_work_requests() {
    // Make sure we are ready to go
    check_null(_pkt_buf,
               "find existing packet buffer");
    check_null(_qp,
               "find existing queue pairs");
    
    // Setup the work requests
    int i, j, k;
    struct ibv_recv_wr* recv_wr_bad;
    
    for(i=0; i<BF_VERBS_NPKTBUF*BF_VERBS_NQP; i++) {
        _pkt_buf[i].wr.wr_id = i;
        _pkt_buf[i].wr.num_sge = 1;
        _sge[i].addr = (uint64_t) _mr_buf + i * _pkt_size_max;
        _sge[i].length = _pkt_size_max;
        
        if( i == 0 ) {
            _pkt_buf[i].wr.sg_list = &(_sge[0]);
        } else {
            _pkt_buf[i].wr.sg_list = _pkt_buf[i-1].wr.sg_list + _pkt_buf[i-1].wr.num_sge;
        }
        
        for(j=0; j<_pkt_buf[i].wr.num_sge; j++) {
            _pkt_buf[i].wr.sg_list[j].lkey = _mr->lkey;
        }
    }
    
    // Link the work requests to receive queue
    for(i=0; i<BF_VERBS_NQP; i++) {
        k = i*BF_VERBS_NPKTBUF;
        for(j=0; j<BF_VERBS_NPKTBUF-1; j++) {
            _pkt_buf[k+j].wr.next = &(_pkt_buf[k+j+1].wr);
        }
        _pkt_buf[k+BF_VERBS_NPKTBUF-1].wr.next = NULL;
        
        // Post work requests to the receive queue
        check_error(ibv_post_recv(_qp[i], &(_pkt_buf[k].wr), &recv_wr_bad),
                    "post work request to receive queue");
    }
}

void Verbs::create_flows() {
    // Setup the flows
    int i;
    struct bf_ibv_flow flow;
    ::memset(&flow, 0, sizeof(struct bf_ibv_flow));
    flow.attr.comp_mask = 0;
    flow.attr.type = IBV_FLOW_ATTR_NORMAL;
    flow.attr.size = sizeof(struct bf_ibv_flow);
    flow.attr.priority = 0;
    flow.attr.num_of_specs = 3;
    flow.attr.port = _port_num;
    flow.attr.flags = 0;
    flow.spec_tcp_udp.type = IBV_FLOW_SPEC_UDP;
    flow.spec_tcp_udp.size = sizeof(flow.spec_tcp_udp);
    flow.spec_ipv4.type = IBV_FLOW_SPEC_IPV4;
    flow.spec_ipv4.size = sizeof(flow.spec_ipv4);
    flow.spec_eth.type = IBV_FLOW_SPEC_ETH;
    flow.spec_eth.size = sizeof(flow.spec_eth);
    
    // Filter on UDP and the port
    flow.spec_tcp_udp.val.dst_port = htons(this->get_port());
    flow.spec_tcp_udp.mask.dst_port = 0xffff;
    
    // Filter on nothing in the IPv4 header
    
    // Filter on the destination MAC address
    this->get_mac_address(&(flow.spec_eth.val.dst_mac[0]));
    ::memset(&(flow.spec_eth.mask.dst_mac[0]), 0xff, 6);
    
    // Create the flows
    _flows = (struct ibv_flow**) ::malloc(BF_VERBS_NQP * sizeof(struct ibv_flow*));
    check_null(_flows,
               "allocate flows");
    ::memset(_flows, 0, BF_VERBS_NQP * sizeof(struct ibv_flow*));
    for(i=0; i<BF_VERBS_NQP; i++) {
        _flows[i] = ibv_create_flow(_qp[i], (struct ibv_flow_attr*) &flow);
        check_null(_flows[i],
                   "create flow");
    }
}

void Verbs::destroy_flows() {
    int failures = 0;
    
    if( _flows ) {
        for(int i=0; i<BF_VERBS_NQP; i++) {
            if( _flows[i] ) {
                if( ibv_destroy_flow(_flows[i]) ) {
                    failures += 1;
                }
            }
        }
        free(_flows);
    }
}

// See comments in header file for details about this function.
int Verbs::release(struct bf_ibv_recv_pkt* recv_pkt) {
    int i;
    struct ibv_recv_wr* recv_wr_bad;
    
    if( !recv_pkt ) {
        return 0;
    }
    
    // Figure out which QP these packets belong to and repost to that QP
    i = recv_pkt->wr.wr_id / BF_VERBS_NPKTBUF;
    return ibv_post_recv(_qp[i], &recv_pkt->wr, &recv_wr_bad);
}

struct bf_ibv_recv_pkt* Verbs::receive(int timeout_ms) {
    int i;
    int num_wce;
    uint64_t wr_id;
    struct pollfd pfd;
    struct ibv_qp_attr qp_attr;
    struct ibv_cq *ev_cq;
    intptr_t ev_cq_ctx;
    struct ibv_wc wc[BF_VERBS_WCBATCH];
    struct bf_ibv_recv_pkt * recv_head = NULL;
    struct ibv_recv_wr * recv_tail = NULL;
    
    // Ensure the queue pairs are in a state suitable for receiving
    for(i=0; i<BF_VERBS_NQP; i++) {
        switch(_qp[i]->state) {
            case IBV_QPS_RESET: // Unexpected, but maybe user reset it
                qp_attr.qp_state = IBV_QPS_INIT;
                qp_attr.port_num = _port_num;
                if( ibv_modify_qp(_qp[i], &qp_attr, IBV_QP_STATE|IBV_QP_PORT) ) {
                    return NULL;
                }
            case IBV_QPS_INIT:
                qp_attr.qp_state = IBV_QPS_RTR;
                if( ibv_modify_qp(_qp[i], &qp_attr, IBV_QP_STATE) ) {
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
    pfd.fd = _cc->fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    
    // poll completion channel's fd with given timeout
    if( ::poll(&pfd, 1, timeout_ms) <= 0 ) {
        // Timeout or error
        return NULL;
    }
    
    // Get the completion event
    if( ibv_get_cq_event(_cc, &ev_cq, (void **)&ev_cq_ctx) ) {
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
        num_wce = ibv_poll_cq(ev_cq, BF_VERBS_WCBATCH, wc);
        if( num_wce < 0 ) {
            return NULL;
        }
        
        // Loop through all work completions
        for(i=0; i<num_wce; i++) {
            wr_id = wc[i].wr_id;
            // Set length to 0 for unsuccessful work requests
            if( wc[i].status != IBV_WC_SUCCESS ) {
                _pkt_buf[wr_id].length = 0;
            } else {
                // Copy byte_len from completion to length of pkt srtuct
                _pkt_buf[wr_id].length = wc[i].byte_len;
            }
            // Add work requests to recv list
            if( !recv_head ) {
                recv_head = &(_pkt_buf[wr_id]);
                recv_tail = &recv_head->wr;
            } else {
                recv_tail->next = &(_pkt_buf[wr_id].wr);
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
