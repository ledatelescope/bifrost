#include <stdio.h>
#include "hashpipe_ibverbs.h"
#include <bifrost/common.h>

#define IBV_UDP_PAYLOAD_OFFSET 42

extern "C" {
struct hashpipe_ibv_context _hibv_ctx = {0};
struct hashpipe_ibv_recv_pkt* _hibv_rpkt; // Current packet chain
struct hashpipe_ibv_recv_pkt* _pkt; // Current packet

BFstatus ibv_init(size_t pkt_size_max) {
    fprintf(stderr, "Configuring IBV socket\n");
    int port = 10000;
    char ifname[IFNAMSIZ] = "ens1f1";
    strncpy(_hibv_ctx.interface_name, ifname, IFNAMSIZ);
    _hibv_ctx.interface_name[IFNAMSIZ-1] = '\0'; // Ensure NUL termination
    _hibv_ctx.send_pkt_num = 1;
    _hibv_ctx.recv_pkt_num = 8192;
    _hibv_ctx.pkt_size_max = pkt_size_max;
    _hibv_ctx.max_flows = 1;
    int ret = hashpipe_ibv_init(&_hibv_ctx);
	if( ret ) {
	    fprintf(stderr, "ERROR: haspipe_ibv_init returned %d\n", ret);
	}

    // Subscribe to RX flow
    ret = hashpipe_ibv_flow(
            &_hibv_ctx,
            0, IBV_FLOW_SPEC_UDP,
            _hibv_ctx.mac, NULL, 0, 0, 0, 0, 0, port);
    if( ret ) {
	    fprintf(stderr, "ERROR: haspipe_ibv_flow returned %d\n", ret);
    }

    return BF_STATUS_SUCCESS;
}

int ibv_recv_packet(uint8_t** pkt_ptr, int flags) {
    // If we don't have a work-request queue on the go,
    // get some new packets.
    if ( _pkt ) {
        _pkt = (struct hashpipe_ibv_recv_pkt *)_pkt->wr.next;
        if ( !_pkt ) {
            hashpipe_ibv_release_pkts(&_hibv_ctx, _hibv_rpkt);
            _hibv_rpkt = NULL;
        }
    }
    while (!_hibv_rpkt) {
        _hibv_rpkt = hashpipe_ibv_recv_pkts(&_hibv_ctx, 1);
        _pkt = _hibv_rpkt;
    }
    // IBV returns Eth/UDP/IP headers. Strip them off here.
    *pkt_ptr = (uint8_t *)_pkt->wr.sg_list->addr + IBV_UDP_PAYLOAD_OFFSET;
    return _pkt->length - IBV_UDP_PAYLOAD_OFFSET;
}

}
