/*
 *  Copyright 2016 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file memory.h
 *  \brief Space- (host/device) aware memory management/copy/set functions
 */

#ifndef BF_MEMORY_H_INCLUDE_GUARD_
#define BF_MEMORY_H_INCLUDE_GUARD_

#include <bifrost/common.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef BF_ALIGNMENT
  #define BF_ALIGNMENT 4096//512
#endif


BFstatus bfMalloc(void** ptr, BFsize size, BFspace space);
BFstatus bfFree(void* ptr, BFspace space);

// TODO: Change this to return status as per the library convention
BFspace bfGetSpace(const void* ptr, BFstatus* status);

// Note: This is sync wrt host but async wrt device
BFstatus bfMemcpy(void*       dst,
                  BFspace     dst_space,
                  const void* src,
                  BFspace     src_space,
                  BFsize      count);
BFstatus bfMemcpy2D(void*       dst,
                    BFsize      dst_stride,
                    BFspace     dst_space,
                    const void* src,
                    BFsize      src_stride,
                    BFspace     src_space,
                    BFsize      width,
                    BFsize      height);
BFstatus bfMemset(void*   ptr,
                  BFspace space,
                  int     value,
                  BFsize  count);
BFstatus bfMemset2D(void*   ptr,
                    BFsize  stride,
                    BFspace space,
                    int     value,
                    BFsize  width,
                    BFsize  height);
BFsize bfGetAlignment();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_MEMORY_H_INCLUDE_GUARD_
