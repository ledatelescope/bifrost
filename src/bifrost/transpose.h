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

#ifndef BF_TRANSPOSE_H_INCLUDE_GUARD_
#define BF_TRANSPOSE_H_INCLUDE_GUARD_

#include <bifrost/common.h>
#include <bifrost/memory.h>

#ifdef __cplusplus
extern "C" {
#endif

BFstatus bfTranspose(void*         dst,
                     BFsize const* dst_strides,
                     void   const* src,
                     BFsize const* src_strides,
                     BFspace       space,
                     BFsize        element_size,
                     BFsize        ndim,
                     BFsize const* src_shape,
                     BFsize const* axes);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_TRANSPOSE_H_INCLUDE_GUARD_
