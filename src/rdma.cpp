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

#include "rdma.hpp"
#include "utils.hpp"

ssize_t BFrdma_impl::send_header(BFoffset    time_tag,
                                  BFsize      header_size,
                                  const void* header,
                                  BFoffset    offset_from_head) {
  BF_ASSERT(header_size <= this->max_message_size(), BF_STATUS_INVALID_SHAPE);
  
  bf_message msg;
  msg.type     = bf_message::MSG_HEADER;
  msg.time_tag = time_tag;
  msg.length   = header_size;
  msg.offset   = offset_from_head;
  
  return _rdma->send(&msg, header);
}

ssize_t BFrdma_impl::send_span(BFarray const* span) {
  size_t span_size = BF_DTYPE_NBYTE(span->dtype);
  for(int i=0; i<span->ndim; i++) {
   span_size *= span->shape[i];
  }
  BF_ASSERT(span_size <= this->max_message_size(), BF_STATUS_INVALID_SHAPE);
  
  bf_message msg;
  msg.type     = bf_message::MSG_SPAN;
  msg.time_tag = 0;
  msg.length   = span_size;
  msg.offset   = 0;
  
  return _rdma->send(&msg, span->data);
}

ssize_t BFrdma_impl::receive(BFoffset* time_tag,
                              BFsize*   header_size,
                              BFoffset* offset_from_head,
                              BFsize*   span_size,
                              void*     contents) {
  bf_message msg;
  ::memset(&msg, 0, sizeof(bf_message));
  ssize_t nrecv;
  nrecv = _rdma->receive(&msg, contents);
  if( msg.type == bf_message::MSG_HEADER) {
    *time_tag         = msg.time_tag;
    *header_size      = msg.length;
    *offset_from_head = msg.offset;
    *span_size        = 0;
  } else {
    *time_tag         = 0;
    *header_size      = 0;
    *offset_from_head = 0;
    *span_size        = msg.length;
    
  }
  
  return nrecv;
}

BFstatus bfRdmaCreate(BFrdma* obj,
                      int     fd,
                      size_t  message_size,
                      int     is_server) {
    BF_ASSERT(message_size <= BF_RDMA_MAXMEM, BF_STATUS_INSUFFICIENT_STORAGE);
    BF_TRY_RETURN_ELSE(*obj = new BFrdma_impl(fd, message_size, is_server),
                       *obj = 0);
}

BFstatus bfRdmaDestroy(BFrdma obj) {
  BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
  delete obj;
  return BF_STATUS_SUCCESS;
}

BFstatus bfRdmaSendHeader(BFrdma      obj,
                          BFoffset    time_tag,
                          BFsize      header_size,
                          const void* header,
                          BFoffset    offset_from_head) {
  BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
  BF_ASSERT(header, BF_STATUS_INVALID_POINTER);
  ssize_t nsent;
  nsent = obj->send_header(time_tag, header_size, header, offset_from_head);
  if( nsent < 0 ) {
    return BF_STATUS_INTERNAL_ERROR;
  }
  return BF_STATUS_SUCCESS;
}

BFstatus bfRdmaSendSpan(BFrdma         obj,
                        BFarray const* span) {
  BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
  BF_ASSERT(span, BF_STATUS_INVALID_POINTER);
  ssize_t nsent;
  nsent = obj->send_span(span);
  if( nsent < 0 ) {
    return BF_STATUS_INTERNAL_ERROR;
  }
  return BF_STATUS_SUCCESS;
}

BFstatus bfRdmaReceive(BFrdma    obj,
                       BFoffset* time_tag,
                       BFsize*   header_size,
                       BFoffset* offset_from_head,
                       BFsize*   span_size,
                       void*     contents) {
  BF_ASSERT(obj, BF_STATUS_INVALID_HANDLE);
  ssize_t nrecv;
  nrecv = obj->receive(time_tag, header_size, offset_from_head, span_size, contents);
  if( nrecv < 0 ) {
    return BF_STATUS_INTERNAL_ERROR;
  }
  return BF_STATUS_SUCCESS;
}
