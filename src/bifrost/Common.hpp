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

#pragma once

#include <bifrost/common.h>

#include <stdexcept>

#define BIFROST_DEFINE_GETTER(type, name, func, ...) \
	inline type name() const { \
		type val; \
		check( func(__VA_ARGS__, &val) ); \
		return val; \
	}

namespace bifrost {

inline void check(BFstatus status) {
	if( status != BF_STATUS_SUCCESS ) {
		// TODO: BFexception?
		throw std::runtime_error(bfGetStatusString(status));
	}
}
inline bool check_failure_only(BFstatus status) {
	BF_DISABLE_DEBUG();
	switch( status ) {
	case BF_STATUS_MEM_ALLOC_FAILED:
	case BF_STATUS_MEM_OP_FAILED:
	case BF_STATUS_DEVICE_ERROR:
	case BF_STATUS_INTERNAL_ERROR:
		check(status);
		return false; // Unreachable
	default: return status == BF_STATUS_SUCCESS;
	}
}

} // namespace bifrost
