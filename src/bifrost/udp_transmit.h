#ifndef BF_UDP_TRANSMIT_H_INCLUDE_GUARD_
#define BF_UDP_TRANSMIT_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BFudptransmit_impl* BFudptransmit;

typedef enum {
	BF_TRANSMIT_CONTINUED, 
	BF_TRANSMIT_INTERRUPTED,
	BF_TRANSMIT_ERROR
} BFudptransmit_status;

BFstatus bfUdpTransmitCreate(BFudptransmit* obj,
                            int           fd,
                            int           core);
BFstatus bfUdpTransmitDestroy(BFudptransmit obj);
BFstatus bfUdpTransmitSend(BFudptransmit obj, char* packet, unsigned int len);
BFstatus bfUdpTransmitSendMany(BFudptransmit obj, char* packets, unsigned int len, unsigned int npackets);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_UDP_TRANSMIT_H_INCLUDE_GUARD_