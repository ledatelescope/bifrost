#ifndef BF_IBVERBS_H_INCLUDE_GUARD_
#define BF_IBVERBS_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

BFstatus ibv_init(size_t pkt_size_max);
int ibv_recv_packet(uint8_t** pkt_ptr, int flags);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_IBVERBS_INCLUDE_GUARD_
