/*
 * Copyright (c) 2019, The Bifrost Authors. All rights reserved.
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

#ifndef BF_PACKET_WRITER_H_INCLUDE_GUARD_
#define BF_PACKET_WRITER_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

#include <bifrost/array.h>

typedef struct BFpacketdescription_impl* BFpacketdescription;

BFstatus bfPacketDescriptionCreate(BFpacketdescription* obj);
BFstatus bfPacketDescriptionDestroy(BFpacketdescription obj);
BFstatus bfPacketDescriptionSetNSrc(BFpacketdescription obj,
                                    int                 nsrc);
BFstatus bfPacketDescriptionSetNChan(BFpacketdescription obj,
                                     int                 nchan);
BFstatus bfPacketDescriptionSetChan0(BFpacketdescription obj,
                                     int                 chan0);
BFstatus bfPacketDescriptionSetTuning(BFpacketdescription obj,
                                      int                 tuning);
BFstatus bfPacketDescriptionSetGain(BFpacketdescription obj,
                                    uint16_t            gain);
BFstatus bfPacketDescriptionSetDecimation(BFpacketdescription obj,
                                          uint16_t            decimation);

typedef struct BFpacketwriter_impl* BFpacketwriter;

BFstatus bfPacketWriterDestroy(BFpacketwriter obj);
BFstatus bfPacketWriterSend(BFpacketwriter      obj, 
                            BFpacketdescription desc,
                            BFoffset            seq,
                            BFoffset            seq_increment,
                            BFoffset            src,
                            BFoffset            src_increment,
                            BFarray const*      in);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_PACKET_WRITER_H_INCLUDE_GUARD_
