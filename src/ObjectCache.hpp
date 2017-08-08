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

#include <map>
#include <deque>
#include <stdexcept>
#include <algorithm>

// Simple cache using LRU discard policy
template<typename KeyType, typename ValueType>
class ObjectCache {
public:
	typedef KeyType   key_type;
	typedef ValueType value_type;
private:
	typedef std::map<key_type,value_type> object_map;
	typedef std::deque<key_type>          key_rank;
	typedef typename key_rank::iterator   rank_iterator;
	object_map _objects;
	key_rank   _ranked_keys;
	size_t     _capacity;
	
	inline void discard_old(size_t n=0) {
		if( n > _capacity ) {
			throw std::runtime_error("Insufficient capacity in cache");
		}
		while( _objects.size() > _capacity-n ) {
			key_type discard_key = _ranked_keys.back();
			_ranked_keys.pop_back();
			_objects.erase(discard_key);
		}
	}
public:
	inline ObjectCache(size_t capacity=8) : _capacity(capacity) {}
	inline void resize(size_t capacity) {
		_capacity = capacity;
		this->discard_old();
	}
	inline bool contains(const key_type& k) const {
		return _objects.count(k);
	}
	inline void touch(const key_type& k) {
		if( !this->contains(k) ) {
			throw std::runtime_error("Key not found in cache");
		}
		rank_iterator rank = std::find(_ranked_keys.begin(),
		                               _ranked_keys.end(),
		                               k);
		if( rank != _ranked_keys.begin() ) {
			// Move key to front of ranks
			_ranked_keys.erase(rank);
			_ranked_keys.push_front(k);
		}
	}
	inline value_type& get(const key_type& k) {
		if( !this->contains(k) ) {
			throw std::runtime_error("Key not found in cache");
		}
		this->touch(k);
		return _objects[k];
	}
	inline value_type& insert(const key_type& k, const value_type& v=value_type()) {
		this->discard_old(1);
		_ranked_keys.push_front(k);
		return _objects.insert(std::make_pair(k,v)).first->second;
	}
};
