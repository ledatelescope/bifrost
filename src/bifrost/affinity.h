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

/*! \file affinity.h
 *  \brief CPU core binding/affinity functions
 */

#ifndef BF_AFFINITY_H_INCLUDE_GUARD_
#define BF_AFFINITY_H_INCLUDE_GUARD_

#include <bifrost/common.h>

#ifdef __cplusplus
extern "C" {
#endif

// Note: Pass core=-1 to unbind
BFstatus bfAffinitySetCore(int core);
BFstatus bfAffinityGetCore(int* core);
BFstatus bfAffinitySetOpenMPCores(BFsize     nthread,
                                  const int* thread_cores);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_AFFINITY_H_INCLUDE_GUARD_
