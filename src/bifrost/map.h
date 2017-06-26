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

/*! \file map.h
 *  \brief Defines a general elementwise array computation function
 */

#ifndef BF_MAP_H_INCLUDE_GUARD_
#define BF_MAP_H_INCLUDE_GUARD_

#include <bifrost/common.h>
#include <bifrost/array.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \p bfMap applies a user-defined indexing and transformation function to a
 *     set of arrays.
 *
 *  \param ndim        The number of dimensions of the computation
 *  \param shape       The shape of the computation; an array of length \p ndim
 *  \param axis_names  The name by which each dimension can be referenced; an array of length \p ndim, or NULL
 *  \param narg        The number of BFarrays to operate on
 *  \param args        The BFarrays to operate on; an array of length \p narg
 *  \param arg_names   The names by which each BFarray can be referenced; an array of length \p narg
 *  \param func        The function to apply to the arrays; a string containing executable code
 *  \param block_shape The 2D shape of the thread block (y,x) with which the kernel is launched.
 *                       This is a performance tuning parameter.
 *                       If NULL, a heuristic is used to select the block shape.
 *                       Changes to this parameter do _not_ require re-compilation of the kernel.
 *  \param block_axes  The 2 computation axes to which the thread block (y,x) is mapped.
 *                        This is a performance tuning parameter.
 *                        If NULL, a heuristic is used to select the block axes.
 *                        Values may be negative for reverse indexing.
 *                        Changes to this parameter _do_ require re-compilation of the kernel.
 *  \return One of the following error codes: \n
 *  \p BF_STATUS_SUCCESS, \p BF_STATUS_INVALID_SPACE,
 *  \p BF_STATUS_INVALID_POINTER, \p BF_STATUS_INVALID_STRIDE,
 *  \p BF_STATUS_UNSUPPORTED_DTYPE, \p BF_STATUS_INVALID_ARGUMENT,
 *  \p BF_STATUS_DEVICE_ERROR
 *  \note The string \p func must be valid C++11 syntax suitable for execution
 *        as CUDA device code. Examples:\n
 *        \code{.cpp} "c(i,j,k) = a(i,k) + b" // Using axis_names = {"i", "j", "k"}\endcode
 *        \code{.cpp} "z(_) = x(_) * y(_ - y.shape()/2)"    // Using the built-in index array "_"\endcode
 *        \code{.cpp} "out(i) = in((i + shift) % in.shape(0))" // Using the shape of an array\endcode
 *        Broadcasting, negative indexing, and complex math are also supported.
 *  \note Any BFarrays that are immutable, have shape=[1] and are accessible
 *        from system memory are treated as scalars, and must be accessed as,
 *         e.g., "b" not "b(0)".
 *  \note While this function is very flexible, it does not guarantee efficient
 *          computation. E.g., using it for transpose operations is likely to
 *          be much less efficient than using the dedicated bfTranspose function.
 */
BFstatus bfMap(int                  ndim,
               long const*          shape,
               char const*const*    axis_names,
               int                  narg,
               BFarray const*const* args,
               char const*const*    arg_names,
               char const*          func,
               int const*           block_shape, // Must be array of length 2, or NULL
               int const*           block_axes); // Must be array of length 2, or NULL

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_MAP_H_INCLUDE_GUARD_
