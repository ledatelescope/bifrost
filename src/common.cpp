/*
 * Copyright (c) 2016, The Bifrost Authors. All rights reserved.
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#include <bifrost/common.h>

#include <sstream>

const char* bfGetStatusString(BFstatus status) {
#define BF_STATUS_STRING_CASE(x) case x: return #x;
	switch( status ) {
		BF_STATUS_STRING_CASE(BF_STATUS_SUCCESS);
		BF_STATUS_STRING_CASE(BF_STATUS_END_OF_DATA);
		BF_STATUS_STRING_CASE(BF_STATUS_INVALID_POINTER);
		BF_STATUS_STRING_CASE(BF_STATUS_INVALID_HANDLE);
		BF_STATUS_STRING_CASE(BF_STATUS_INVALID_ARGUMENT);
		BF_STATUS_STRING_CASE(BF_STATUS_INVALID_STATE);
		BF_STATUS_STRING_CASE(BF_STATUS_INVALID_SPACE);
		BF_STATUS_STRING_CASE(BF_STATUS_INVALID_SHAPE);
		BF_STATUS_STRING_CASE(BF_STATUS_INVALID_STRIDE);
		BF_STATUS_STRING_CASE(BF_STATUS_INVALID_DTYPE);
		BF_STATUS_STRING_CASE(BF_STATUS_MEM_ALLOC_FAILED);
		BF_STATUS_STRING_CASE(BF_STATUS_MEM_OP_FAILED);
		BF_STATUS_STRING_CASE(BF_STATUS_UNSUPPORTED);
		BF_STATUS_STRING_CASE(BF_STATUS_UNSUPPORTED_SPACE);
		BF_STATUS_STRING_CASE(BF_STATUS_UNSUPPORTED_SHAPE);
		BF_STATUS_STRING_CASE(BF_STATUS_UNSUPPORTED_STRIDE);
		BF_STATUS_STRING_CASE(BF_STATUS_UNSUPPORTED_DTYPE);
		BF_STATUS_STRING_CASE(BF_STATUS_FAILED_TO_CONVERGE);
		BF_STATUS_STRING_CASE(BF_STATUS_INSUFFICIENT_STORAGE);
		BF_STATUS_STRING_CASE(BF_STATUS_DEVICE_ERROR);
		BF_STATUS_STRING_CASE(BF_STATUS_INTERNAL_ERROR);
	default: {
		std::stringstream ss;
		ss << "Invalid status code: " << status;
		return ss.str().c_str();
	}
	}
#undef BF_STATUS_STRING_CASE
}

static thread_local bool g_debug_enabled = true;

BFbool bfGetDebugEnabled() {
#if BF_DEBUG
	return g_debug_enabled;
#else
	return false;
#endif
}
BFstatus bfSetDebugEnabled(BFbool b) {
#if !BF_DEBUG
	return BF_STATUS_INVALID_STATE;
#else
	g_debug_enabled = b;
	return BF_STATUS_SUCCESS;
#endif
}
BFbool bfGetCudaEnabled() {
#ifdef BF_CUDA_ENABLED
	return BF_CUDA_ENABLED;
#else
	return false;
#endif
}
