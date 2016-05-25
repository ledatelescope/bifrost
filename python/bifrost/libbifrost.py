
#  Copyright 2016 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# This file provides a direct interface to libbifrost.so

# PYCLIBRARY ISSUE: Passing the wrong handle type to a function gives this meaningless error:
#  ArgumentError: argument 1: <type 'exceptions.TypeError'>: expected LP_s instance instead of LP_s
#  E.g., _bf.RingSequenceGetName(<BFspan>) [should be <BFsequence>]

def _load_bifrost_lib():
	import os
	# TODO: Keep these up-to-date
	headers      = [   "common.h",
					 "affinity.h",
					   "memory.h",
						 "ring.h",
					"transpose.h"]
	library_name = "libbifrost.so"
	api_prefix   = "bf"
	header_paths = ["/usr/local/include/bifrost",
					"../src/bifrost"] # TODO: Remove this one?
	include_env  = 'BIFROST_INCLUDE_PATH'
	# PYCLIBRARY ISSUE
	# TODO: Would it make sense to build this into PyCLibrary?
	library_env  = 'LD_LIBRARY_PATH'
	home_dir     = os.path.expanduser("~")
	parser_cache = os.path.join(home_dir, ".cache/bifrost.parse")
	
	def _get_env_paths(env):
		paths = os.getenv(env)
		if paths is None:
			return []
		return [p for p in paths.split(':')
				if len(p.strip())]
		
	import pyclibrary
	from pyclibrary import CParser, CLibrary
	
	import ctypes
	# PYCLIBRARY ISSUE Should these be built in? Optional extra?
	# Note: This is needed because pyclibrary starts with only
	#         the fundamental types (short, int, float etc.).
	#extra_types = {}
	#extra_types = {'uint64_t': ctypes.c_uint64}
	extra_types = {
		' uint8_t': ctypes.c_uint8,
		'  int8_t': ctypes.c_int8,
		'uint16_t': ctypes.c_uint16,
		' int16_t': ctypes.c_int16,
		'uint32_t': ctypes.c_uint32,
		' int32_t': ctypes.c_int32,
		'uint64_t': ctypes.c_uint64,
		' int64_t': ctypes.c_int64
	}
	
	try:
		pyclibrary.auto_init(extra_types=extra_types)
	except RuntimeError:
		pass # WAR for annoying "Can only initialise the parser once"
	header_paths += _get_env_paths(include_env)
	valid_header_paths = [p for p in header_paths if os.path.exists(p)]
	pyclibrary.utils.add_header_locations(valid_header_paths)
	try:
		_parser = CParser(headers, cache=unicode(parser_cache, "utf-8"))
	except AttributeError: # # PYCLIBRARY ISSUE WAR for "'tuple' has no attribute 'endswith'" bug
		raise ValueError("Could not find Bifrost headers.\nSearch paths: "+
						 str(header_paths))
	pyclibrary.utils.add_library_locations(_get_env_paths(library_env))
	lib = CLibrary(library_name, _parser, prefix=api_prefix)
	return lib

_bf = _load_bifrost_lib() # Internal access to library
bf = _bf                  # External access to library

# Internal helper functions below

def _array(typ, size_or_vals):
	from pyclibrary import build_array
	try:
		_ = iter(size_or_vals)
		vals = size_or_vals
		return build_array(_bf, typ, size=len(vals), vals=vals)
	except TypeError:
		size = size_or_vals
		return build_array(_bf, typ, size=size)

def _check(f):
	status, args = f
	if status != _bf.BF_STATUS_SUCCESS:
		if status is None:
			raise RuntimeError("WTF, status is None")
		if status == _bf.BF_STATUS_END_OF_DATA:
			raise StopIteration()
		else:
			status_str, _ = _bf.GetStatusString(status)
			raise RuntimeError(status_str)
	return f

def _get(f, retarg=-1):
	status, args = _check(f)
	return list(args)[retarg]

def _retval(f):
	ret, args = f
	return ret

def _string2space(s):
	lut = {'auto':         _bf.BF_SPACE_AUTO,
	       'system':       _bf.BF_SPACE_SYSTEM,
	       'cuda':         _bf.BF_SPACE_CUDA,
	       'cuda_host':    _bf.BF_SPACE_CUDA_HOST,
	       'cuda_managed': _bf.BF_SPACE_CUDA_MANAGED}
	if s not in lut:
		raise KeyError("Invalid space '"+str(s)+"'.\nValid spaces: "+str(lut.keys()))
	return lut[s]
def _space2string(i):
	return {_bf.BF_SPACE_AUTO:         'auto',
	        _bf.BF_SPACE_SYSTEM:       'system',
	        _bf.BF_SPACE_CUDA:         'cuda',
	        _bf.BF_SPACE_CUDA_HOST:    'cuda_host',
	        _bf.BF_SPACE_CUDA_MANAGED: 'cuda_managed'}[i]
