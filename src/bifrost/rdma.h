/*
 * Copyright (c) 2022, The Bifrost Authors. All rights reserved.
 * Copyright (c) 2022, The University of New Mexico. All rights reserved.
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

#ifndef BF_RDMA_H_INCLUDE_GUARD_
#define BF_RDMA_H_INCLUDE_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

#include <bifrost/array.h>

typedef struct BFrdma_impl* BFrdma;

BFstatus bfRdmaCreate(BFrdma* obj,
                      int     fd,
                      size_t  message_size,
                      int     is_server);
BFstatus bfRdmaDestroy(BFrdma obj);
BFstatus bfRdmaSendHeader(BFrdma obj,
                          BFoffset    time_tag,
                          BFsize      header_size,
                          const void* header,
                          BFoffset    offset_from_head);
BFstatus bfRdmaSendSpan(BFrdma         obj,
                        BFarray const* span);
BFstatus bfRdmaReceive(BFrdma    obj,
                       BFoffset* time_tag,
                       BFsize*   header_size,
                       BFoffset* offset_from_head,
                       BFsize*   span_size,
                       void*     contents);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_RDMA_H_INCLUDE_GUARD_
