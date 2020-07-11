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

#include "ib_verbs.hpp"

void Verbs::create_context() {
    int d, p, g;
    int ndev, found;
    struct ibv_context* ibv_ctx = NULL;
    struct ibv_device_attr ibv_dev_attr;
    struct ibv_port_attr ibv_port_attr;
    union  ibv_gid ibv_gid;
    
    // Get the interface ID
    found = 0;
    uint64_t iid = this->get_interface_id();
    
    // Find the right device
    /* Query all devices */
    _dev_list = ibv_get_device_list(&ndev);
    check_null(_dev_list,
               "ibv_get_device_list");
    
    /* Interogate */
    for(d=0; d<ndev; d++) {
        ibv_ctx = ibv_open_device(_dev_list[d]);
        check_null(ibv_ctx,
                   "cannot open device");
        
        check_error(ibv_query_device(ibv_ctx, &ibv_dev_attr),
                    "cannot query device");
                    
        /* Loop through the ports on the device */
        for(p=1; p<=ibv_dev_attr.phys_port_cnt; p++) {
            check_error(ibv_query_port(ibv_ctx, p, &ibv_port_attr),
                        "cannot query port");
            
            /* Loop through GIDs of the port on the device */
            for(g=0; g<ibv_port_attr.gid_tbl_len; g++) {
                check_error(ibv_query_gid(ibv_ctx, p, g, &ibv_gid),
                            "cannot query gid on port");
                
                /* Did we find a match? */
                if( (ibv_gid.global.subnet_prefix == 0x80feUL) \
                   && (ibv_gid.global.interface_id  == iid) ) {
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
            *_ctx = *ibv_ctx;
            ::memcpy(&_dev_attr, &ibv_dev_attr, sizeof(ibv_device_attr));
            _port_num = p;
            break;
        } else {
            check_error(ibv_close_device(ibv_ctx),
                        "cannot close device");
        }
    }
    
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
    
    if( _dev_list ) {
        ibv_free_device_list(_dev_list);
    }
}

void Verbs::create_buffers() {
    // Setup the protected domain
    _pd = ibv_alloc_pd(_ctx);
    
    // Create the buffers, the scatter/gather entries, and the memory region
    _pkt_buf = (bf_ibv_recv_pkt*) ::malloc(BF_VERBS_NPKTBUF*BF_VERBS_NQP * sizeof(bf_ibv_recv_pkt));
    check_null(_pkt_buf, 
               "cannot allocate receive packet buffer");
    ::memset(_pkt_buf, 0, BF_VERBS_NPKTBUF*BF_VERBS_NQP * sizeof(bf_ibv_recv_pkt));
    _sge = (struct ibv_sge*) ::malloc(BF_VERBS_NPKTBUF*BF_VERBS_NQP * sizeof(struct ibv_sge));
    check_null(_sge,
               "cannot allocate scatter/gather entries");
    ::memset(_sge, 0, BF_VERBS_NPKTBUF*BF_VERBS_NQP * sizeof(struct ibv_sge));
    _mr_size = (size_t) BF_VERBS_NPKTBUF*BF_VERBS_NQP * _pkt_size_max;
    _mr_buf = (uint8_t *) ::malloc(_mr_size);
    check_null(_mr_buf,
               "cannot allocate memory region buffer");
    ::memset(_mr_buf, 0, _mr_size);
    _mr = ibv_reg_mr(_pd, _mr_buf, _mr_size, IBV_ACCESS_LOCAL_WRITE);
    check_null(_mr,
               "cannot register memory region");
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
               "cannot create completion channel");
    int flags = ::fcntl(_cc->fd, F_GETFL);
    ::fcntl(_cc->fd, F_SETFL, flags | O_NONBLOCK);
    
    // Setup the completion queues
    _cq = (struct ibv_cq**) ::malloc(BF_VERBS_NQP * sizeof(struct ibv_cq*));
    check_null(_cq,
               "cannot allocate completion queues");
    
    for(i=0; i<BF_VERBS_NQP; i++) {
        _cq[i] = ibv_create_cq(_ctx, BF_VERBS_NPKTBUF, (void *)(uintptr_t)i, _cc, 0);
        check_null(_cq[i],
                   "cannot crete competion queue");
        
        // Request notifications before any receive completion can be created.
        // Do NOT restrict to solicited-only completions for receive.
        check_error(ibv_req_notify_cq(_cq[i], 0),
                    "cannot change completion queue request notifications");
    }
    
    // Setup the queue pairs
    struct ibv_qp_init_attr qp_init;
    ::memset(&qp_init, 0, sizeof(struct ibv_qp_init_attr));
    qp_init.qp_context = NULL;
    qp_init.srq = NULL;
    qp_init.cap.max_send_wr = 1;
    qp_init.cap.max_recv_wr = BF_VERBS_NPKTBUF;
    qp_init.cap.max_send_sge = 1;
    qp_init.cap.max_recv_sge = 1;
    qp_init.cap.max_inline_data = 0;
    qp_init.qp_type = IBV_QPT_RAW_PACKET;
    qp_init.sq_sig_all = 0;
    
    _qp = (struct ibv_qp**) ::malloc(BF_VERBS_NQP*sizeof(struct ibv_qp*));
    check_null(_qp,
               "cannot allocate queue pairs");
    for(i=0; i<BF_VERBS_NQP; i++) {
        qp_init.recv_cq = _cq[i];
        _qp[i] = ibv_create_qp(_pd, &qp_init);
        check_null(_qp[i],
                   "cannot create queue pair");
        
        // Transition QP to INIT state
        struct ibv_qp_attr qp_attr;
        ::memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
        qp_attr.qp_state = IBV_QPS_INIT;
        qp_attr.port_num = _port_num;
        
        check_error(ibv_modify_qp(_qp[i], &qp_attr, IBV_QP_STATE|IBV_QP_PORT),
                    "cannot modify queue pair state");
    }

}

void Verbs::destroy_queues() {
    int failures = 0;
    
    if( _qp ) {
        for(int i=0; i<BF_VERBS_NQP; i++) {
            if( ibv_destroy_qp(_qp[i]) ) {
                failures += 1;
            }
        }
        free(_qp);
    }
    
    if( _cq ) {
        for(int i=0; i<BF_VERBS_NQP; i++) {
            if( ibv_destroy_cq(_cq[i]) ) {
                failures += 1;
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
               "packet buffer has not been created");
    check_null(_qp,
               "queue pairs have not been created");
    
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
            _pkt_buf[k+j].wr.next = &_pkt_buf[k+j+1].wr;
        }
        _pkt_buf[k+j].wr.next = NULL;
        
        // Post work requests to the receive queue
        check_error(ibv_post_recv(_qp[i], &_pkt_buf[k].wr, &recv_wr_bad),
                    "cannot post work request to receive queue");
    }
}

void Verbs::create_flows() {
    // Setup the flows
    _flows = (struct ibv_flow**) ::malloc(_nflows*BF_VERBS_NQP * sizeof(struct ibv_flow*));
    check_null(_flows,
               "cannot allocate flows");
    
    int i, j;
    struct bf_ibv_flow flow;
    ::memset(&flow, 0, sizeof(struct bf_ibv_flow));
    flow.attr.comp_mask = 0;
    flow.attr.type = IBV_FLOW_ATTR_NORMAL;
    flow.attr.size = sizeof(flow.attr);
    flow.attr.priority = 0;
    flow.attr.num_of_specs = 0;
    flow.attr.port = _port_num;
    flow.attr.flags = 0;
    flow.spec_eth.type = IBV_FLOW_SPEC_ETH;
    flow.spec_eth.size = sizeof(flow.spec_eth);
    flow.spec_ipv4.type = IBV_FLOW_SPEC_IPV4;
    flow.spec_ipv4.size = sizeof(flow.spec_ipv4);
    flow.spec_tcp_udp.type = IBV_FLOW_SPEC_UDP;
    flow.spec_tcp_udp.size = sizeof(flow.spec_tcp_udp);
    
    // Filter on UDP and the port
    flow.attr.size += sizeof(struct ibv_flow_spec_tcp_udp);
    flow.attr.num_of_specs++;
    flow.spec_tcp_udp.val.dst_port = htobe16(this->get_port());
    flow.spec_tcp_udp.mask.dst_port = 0xffff;
    
    // Filter on the destination MAC address
    flow.attr.size += sizeof(struct ibv_flow_spec_eth);
    flow.attr.num_of_specs++;
    this->get_mac((uint8_t*) &(flow.spec_eth.val.dst_mac));
    ::memset(flow.spec_eth.mask.dst_mac, 0xff, 6);
    
    // Create the flows
    for(i=0; i<BF_VERBS_NQP; i++) {
        j = i*1 + 0;
        _flows[j] = ibv_create_flow(_qp[i], (struct ibv_flow_attr*) &flow);
        check_null(_flows[j],
                   "cannot create flow");
    }
}

void Verbs::destroy_flows() {
    int failures = 0;
    
    if( _flows ) {
        for(int i=0; i<_nflows; i++) {
            if( ibv_destroy_flow(_flows[i]) ) {
                failures += 1;
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

