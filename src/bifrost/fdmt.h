/*
 * Copyright (c) 2016, The Bifrost Authors. All rights reserved.
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

/*! \file fdmt.h
 *  \brief Defines functions for computing the Fast Dispersion Measure
 *         Transform (FDMT) of Zackay and Ofek (2014),
 *         https://arxiv.org/abs/1411.5373
 */

// ***TODO: Replace BFsize with long/size_t/int as appropriate

#ifndef BF_FDMT_H_INCLUDE_GUARD_
#define BF_FDMT_H_INCLUDE_GUARD_

#include <bifrost/common.h>
#include <bifrost/memory.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BFfdmt_impl* BFfdmt;

BFstatus bfFdmtCreate(BFfdmt* plan);

/*! \p bfFdmtInit initialises a FDMT plan.
 *
 *  \param plan              The FDMT plan to intialize
 *  \param nchan             The number of frequency channels
 *  \param max_delay         The number of delays to compute (the delay in time-samples across the band)
 *  \param f0                The frequency of the first channel (units must match \p df)
 *  \param df                The frequency step between channels (units must match \p f0; may be negative)
 *  \param exponent          The exponent of the delay as a function of frequency (e.g., -2.0 for interstellar dispersion)
 *  \param space             The memory space in which the computation will take place
 *  \param plan_storage      Pointer to memory storage for internal plan data
 *  \param plan_storage_size Pointer to the size in bytes of the memory at \p plan_storage
 *  \return One of the following error codes: \n
 *  \p BF_STATUS_SUCCESS, \p BF_STATUS_INVALID_HANDLE,
 *  \p BF_STATUS_INVALID_POINTER, \p BF_STATUS_INVALID_SPACE,
 *  \p BF_STATUS_INVALID_SHAPE, \p BF_STATUS_INVALID_STRIDE,
 *  \p BF_STATUS_UNSUPPORTED_DTYPE, \p BF_STATUS_UNSUPPORTED_STRIDE,
 *  \p BF_STATUS_INVALID_ARGUMENT, BF_STATUS_INSUFFICIENT_STORAGE,
 *  \p BF_STATUS_MEM_OP_FAILED,
 *  \p BF_STATUS_DEVICE_ERROR, \p BF_STATUS_INTERNAL_ERROR
 *  \note If \p plan_storage == NULL and \p plan_storage_size == NULL, the
 *        plan will manage memory itself.
 *  \note If \p plan_storage == NULL and \p plan_storage_size != NULL, the
 *        function will return the required size in \p *plan_storage_size
 *        and do nothing else.
 *  \note If \p plan_storage != NULL and \p plan_storage_size != NULL, the
 *        function will use this user-defined memory to store the plan.
 *        The memory must exist for the lifetime of the plan.
 */
BFstatus bfFdmtInit(BFfdmt  plan,
                    BFsize  nchan,
                    BFsize  max_delay,
                    double  f0,
                    double  df,
                    double  exponent,
                    BFspace space,
                    void*   plan_storage,
                    BFsize* plan_storage_size);
BFstatus bfFdmtSetStream(BFfdmt      plan,
                         void const* stream);

/*! \p bfFdmtExecute executes a FDMT plan.
 *
 *  \param plan              The FDMT plan to execute
 *  \param iarray            The input filterbank of shape [..., ntime, \p nchan]
 *  \param oarray            The output dispersion bank of shape [..., ntime, \p max_delay]
 *  \param negative_delays   If \p true, the function computes delays in the range (-max_delay, 0] instead of [0, max_delay)
 *  \param exec_storage      Pointer to memory storage for temporary execution data
 *  \param exec_storage_size Pointer to the size in bytes of the memory at \p exec_storage
 *  \return One of the following error codes: \n
 *  \p BF_STATUS_SUCCESS, \p BF_STATUS_INVALID_HANDLE,
 *  \p BF_STATUS_INVALID_POINTER, \p BF_STATUS_INVALID_SPACE,
 *  \p BF_STATUS_INVALID_SHAPE, \p BF_STATUS_INVALID_STRIDE,
 *  \p BF_STATUS_UNSUPPORTED_DTYPE, \p BF_STATUS_UNSUPPORTED_STRIDE,
 *  \p BF_STATUS_INVALID_ARGUMENT, BF_STATUS_INSUFFICIENT_STORAGE,
 *  \p BF_STATUS_MEM_OP_FAILED,
 *  \p BF_STATUS_DEVICE_ERROR, \p BF_STATUS_INTERNAL_ERROR
 *  \note If \p exec_storage == NULL and \p exec_storage_size == NULL, the
 *        function will manage memory itself and execute the plan.
 *  \note If \p exec_storage == NULL and \p exec_storage_size != NULL, the
 *        function will return the required size in \p *exec_storage_size
 *        and do nothing else.
 *  \note If \p exec_storage != NULL and \p exec_storage_size != NULL, the
 *        function will use this user-defined memory to execute the plan.
 *        The memory need exist only until the function returns.
 *  \note The output TOA is aligned with the input TOA at the highest frequency,
 *        and the last (or first, if negative_delays is true) \p max_delay time
 *        samples of the computed output array will be incomplete.
 */
BFstatus bfFdmtExecute(BFfdmt         plan,
                       BFarray const* iarray,
                       BFarray const* oarray,
                       BFbool         negative_delays,
                       void*          exec_storage,
                       BFsize*        exec_storage_size);
BFstatus bfFdmtDestroy(BFfdmt plan);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_FDMT_H_INCLUDE_GUARD_
