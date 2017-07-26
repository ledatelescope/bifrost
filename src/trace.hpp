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

#include "cuda.hpp"

#if BF_CUDA_ENABLED
#include <nvToolsExt.h>
#endif

#include <map>
#include <queue>
#include <string>
#include <cstring>

#if BF_TRACE_ENABLED
// Note: __PRETTY_FUNCTION__ is GCC-specific
//       __FUNCSIG__ is the equivalent in MSVC
#define BF_TRACE()                         ScopedTracer _bf_tracer(__PRETTY_FUNCTION__)
#define BF_TRACE_NAME(name)                ScopedTracer _bf_tracer(name)
#define BF_TRACE_STREAM(stream)            ScopedTracer _bf_stream_tracer(__PRETTY_FUNCTION__, stream)
#define BF_TRACE_NAME_STREAM(name, stream) ScopedTracer _bf_stream_tracer(name, stream)
#else // not BF_TRACE_ENABLED
#define BF_TRACE()
#define BF_TRACE_NAME(name)
#define BF_TRACE_STREAM(stream)
#define BF_TRACE_NAME_STREAM(name, stream)
#endif // BF_TRACE_ENABLED

namespace profile_detail {
inline unsigned simple_hash(const char* c) {
	enum { M = 33 };
	unsigned hash = 5381;
	while( *c ) { hash = hash*M + *c++; }
	return hash;
}
inline uint32_t get_color(unsigned hash) {
	const uint32_t colors[] = {
		0x00aedb, 0xa200ff, 0xf47835, 0xd41243, 0x8ec127,
		0xffb3ba, 0xffdfba, 0xffffba, 0xbaffc9, 0xbae1ff,
		0xbbcbdb, 0x9ebd9e, 0xdd855c, 0xf1e8ca, 0x745151,
		0x2e4045, 0x83adb5, 0xc7bbc9, 0x5e3c58, 0xbfb5b2,
		0xff77aa, 0xaaff77, 0x77aaff, 0xffffff, 0x000000
	};
	const int ncolor = sizeof(colors) / sizeof(uint32_t);
	return colors[hash % ncolor];
}
} // namespace profile_detail

#if BF_CUDA_ENABLED

namespace nvtx {

class AsyncTracer {
	cudaStream_t          _stream;
	nvtxRangeId_t         _id;
	std::string           _msg;
	nvtxEventAttributes_t _attrs;
	inline static void range_start_callback(cudaStream_t stream, cudaError_t status, void* userData) {
		AsyncTracer* range = (AsyncTracer*)userData;
		range->_id = nvtxRangeStartEx(&range->_attrs);
	}
	inline static void range_end_callback(cudaStream_t stream, cudaError_t status, void* userData) {
		AsyncTracer* range = (AsyncTracer*)userData;
		nvtxRangeEnd(range->_id);
		range->_id = 0;
		delete range;
	}
public:
	inline AsyncTracer(cudaStream_t stream) : _stream(stream), _id(0), _attrs() {}
	inline void start(const char* msg, uint32_t color, uint32_t category) {
		_msg = msg;
		_attrs.version       = NVTX_VERSION;
		_attrs.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
		_attrs.colorType     = NVTX_COLOR_ARGB;
		_attrs.color         = color;
		_attrs.messageType   = NVTX_MESSAGE_TYPE_ASCII;
		_attrs.message.ascii = _msg.c_str();
		_attrs.category      = category;
		cudaStreamAddCallback(_stream, range_start_callback, (void*)this, 0);
	}
	inline void end() {
		cudaStreamAddCallback(_stream, range_end_callback, (void*)this, 0);
	}
};

typedef std::map<cudaStream_t,std::queue<AsyncTracer*> > TracerStreamMap;
extern thread_local TracerStreamMap g_nvtx_streams;

} // namespace nvtx

#endif // BF_CUDA_ENABLED

class ScopedTracer {
	std::string _name;
	uint32_t    _color;
	uint32_t    _category;
#if BF_CUDA_ENABLED
	cudaStream_t _stream;
#endif
	// Not copy-assignable
	ScopedTracer(ScopedTracer const& );
	ScopedTracer& operator=(ScopedTracer const& );
#if BF_CUDA_ENABLED
	void build_attrs(nvtxEventAttributes_t* attrs) {
		::memset(attrs, 0, sizeof(*attrs));
		attrs->version       = NVTX_VERSION;
		attrs->size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
		attrs->colorType     = NVTX_COLOR_ARGB;
		attrs->color         = _color;
		attrs->messageType   = NVTX_MESSAGE_TYPE_ASCII;
		attrs->message.ascii = _name.c_str();
		attrs->category      = _category;
	}
#endif
public:
#if BF_CUDA_ENABLED
	inline ScopedTracer(std::string name, cudaStream_t stream=0)
		: _name(name),
		  _color(profile_detail::get_color(profile_detail::simple_hash(name.c_str()))),
		  _category(123),
		  _stream(stream) {
		if( _stream ) {
			nvtx::g_nvtx_streams[_stream].push(new nvtx::AsyncTracer(stream));
			nvtx::g_nvtx_streams[_stream].back()->start(("[G]"+_name).c_str(),
			                                            _color, _category);
		} else {
			nvtxEventAttributes_t attrs;
			this->build_attrs(&attrs);
			nvtxRangePushEx(&attrs);
		}
	}
	inline ~ScopedTracer() {
		if( _stream ) {
			nvtx::g_nvtx_streams[_stream].front()->end();
			nvtx::g_nvtx_streams[_stream].pop();
		} else {
			nvtxRangePop();
		}
	}
#else
	inline explicit ScopedTracer(std::string name)
		: _name(name) {}
	inline ~ScopedTracer() {}
#endif
	
};
