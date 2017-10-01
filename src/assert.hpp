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

#pragma once

#include <bifrost/common.h>

#include <stdexcept>

class BFexception : public std::runtime_error {
	BFstatus _status;
public:
	BFexception(BFstatus stat, const char* msg="")
		: std::runtime_error(std::string(bfGetStatusString(stat))+
		                     ": "+msg),
		  _status(stat) {}
	BFstatus status() const { return _status; }
};

namespace {
inline bool should_report_error(BFstatus err) {
	return (err != BF_STATUS_END_OF_DATA &&
	        err != BF_STATUS_WOULD_BLOCK);
}
}

#include <iostream>
using std::cout;
using std::endl;
#if defined(BF_DEBUG) && BF_DEBUG
	#define BF_REPORT_ERROR(err) do { \
		if( bfGetDebugEnabled() && \
		    should_report_error(err) ) { \
			std::cerr << __FILE__ << ":" << __LINE__ \
			          << " error " << err << ": " \
			          << bfGetStatusString(err) << std::endl; \
		} \
		} while(0)
	#define BF_DEBUG_PRINT(x) do { \
		if( bfGetDebugEnabled() ) { \
			std::cout << __FILE__ << ":" << __LINE__ \
			          << " " #x << " = " << (x) << std::endl; \
		} \
		} while(0)
	#define BF_REPORT_PREDFAIL(pred, err) do { \
		if( bfGetDebugEnabled() && \
		    should_report_error(err) ) { \
			std::cerr << __FILE__ << ":" << __LINE__ \
			          << " Condition failed: " \
			          << #pred << std::endl; \
		} \
		} while(0)
#else
	#define BF_REPORT_ERROR(err)
	#define BF_DEBUG_PRINT(x)
	#define BF_REPORT_PREDFAIL(pred, err)
#endif // BF_DEBUG
#define BF_REPORT_INTERNAL_ERROR(msg) do { \
		std::cerr << __FILE__ << ":" << __LINE__ \
		          << " internal error: " \
		          << msg << std::endl; \
	} while(0)

#define BF_FAIL(msg, err) do { \
		BF_REPORT_PREDFAIL(msg, err); \
		BF_REPORT_ERROR(err); \
		return (err); \
	} while(0)
#define BF_FAIL_EXCEPTION(msg, err) do { \
		BF_REPORT_PREDFAIL(msg, err); \
		BF_REPORT_ERROR(err); \
		throw BFexception(err); \
	} while(0)
#define BF_ASSERT(pred, err) do { \
		if( !(pred) ) { \
			BF_REPORT_PREDFAIL(pred, err); \
			BF_REPORT_ERROR(err); \
			return (err); \
		} \
	} while(0)
#define BF_TRY_ELSE(code, onfail) do { \
		try { code; } \
		catch( BFexception const& err ) { \
			onfail; \
			BF_REPORT_ERROR(err.status()); \
			return err.status(); \
		} \
		catch(std::bad_alloc const& err) { \
			onfail; \
			BF_REPORT_ERROR(BF_STATUS_MEM_ALLOC_FAILED); \
			return BF_STATUS_MEM_ALLOC_FAILED; \
		} \
		catch(std::exception const& err) { \
			onfail; \
			BF_REPORT_INTERNAL_ERROR(err.what()); \
			return BF_STATUS_INTERNAL_ERROR; \
		} \
		catch(...) { \
			onfail; \
			BF_REPORT_INTERNAL_ERROR("FOREIGN EXCEPTION"); \
			return BF_STATUS_INTERNAL_ERROR; \
		} \
	} while(0)
#define BF_NO_OP (void)0
#define BF_TRY(code) BF_TRY_ELSE(code, BF_NO_OP)
#define BF_TRY_RETURN(code) BF_TRY(code); return BF_STATUS_SUCCESS
#define BF_TRY_RETURN_ELSE(code, onfail) BF_TRY_ELSE(code, onfail); return BF_STATUS_SUCCESS

#define BF_ASSERT_EXCEPTION(pred, err) \
	do { \
		if( !(pred) ) { \
			BF_REPORT_PREDFAIL(pred, err); \
			BF_REPORT_ERROR(err); \
			throw BFexception(err); \
		} \
	} while(0)

#define BF_CHECK(call) do { \
	BFstatus status = call; \
	if( status != BF_STATUS_SUCCESS ) { \
		BF_REPORT_ERROR(status); \
		return status; \
	} \
} while(0)

#define BF_CHECK_EXCEPTION(call) do { \
	BFstatus status = call; \
	if( status != BF_STATUS_SUCCESS ) { \
		BF_REPORT_ERROR(status); \
		throw BFexception(status); \
	} \
} while(0)

class NoDebugScope {
	bool _enabled_before;
public:
	NoDebugScope(NoDebugScope const&) = delete;
	NoDebugScope& operator=(NoDebugScope const&) = delete;
	NoDebugScope() : _enabled_before(bfGetDebugEnabled()) {
		bfSetDebugEnabled(false);
	}
	~NoDebugScope() {
		bfSetDebugEnabled(_enabled_before);
	}
};
// Disables debug-printing in the current scope
#define BF_DISABLE_DEBUG() NoDebugScope _bf_no_debug_scope
