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

/*
  A helper class for managing single-allocation workspaces
  E.g.,
  float* array1;
  short* array2;
  Workspace workspace;
  workspace.reserve(100, &array1);
  workspace.reserve(10, &array2);
  void* workspace_ptr = workspace.commit(malloc(workspace.size()));
  // Pointers array1 and array2 now reference parts of the allocation
  free(workspace_ptr);
 */

#pragma once

#include <vector>

class Workspace {
	size_t _size;
	size_t _default_alignment_bytes;
	typedef std::pair<void**,size_t> Reservation; // (&ptr,offset) pair
	std::vector<Reservation> _reservations;
public:
	enum { DEFAULT_ALIGNMENT_BYTES = 4096 };
	inline Workspace(size_t default_alignment_bytes=DEFAULT_ALIGNMENT_BYTES)
		: _size(0),
		  _default_alignment_bytes(default_alignment_bytes) {}
	template<typename T>
	inline size_t reserve(size_t count, T** ptr, size_t alignment_bytes=0) {
		if( !alignment_bytes ) {
			alignment_bytes = _default_alignment_bytes;
		}
		_size = round_up(_size, alignment_bytes);
		ptrdiff_t offset = _size;
		_size += count*sizeof(T);
		Reservation r((void**)ptr, offset);
		_reservations.push_back(r);
		return offset;
	}
	inline size_t size() const { return _size; }
	inline void*  commit(void* base_ptr) {
		// Write the absolute pointer for each reservation
		while( _reservations.size() ) {
			Reservation r = _reservations.back();
			_reservations.pop_back();
			*r.first = (void*)((char*)base_ptr + r.second);
		}
		_size = 0;
		return base_ptr;
	}
	inline void set_default_alignment_bytes(size_t b) {
		_default_alignment_bytes = b;
	}
};
