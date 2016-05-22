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
		BF_STATUS_STRING_CASE(BF_STATUS_MEM_ALLOC_FAILED);
		BF_STATUS_STRING_CASE(BF_STATUS_MEM_OP_FAILED);
		BF_STATUS_STRING_CASE(BF_STATUS_UNSUPPORTED);
		BF_STATUS_STRING_CASE(BF_STATUS_FAILED_TO_CONVERGE);
		BF_STATUS_STRING_CASE(BF_STATUS_INTERNAL_ERROR);
	default: {
		std::stringstream ss;
		ss << "Invalid status code: " << status;
		return ss.str().c_str();
	}
	}
#undef BF_STATUS_STRING_CASE
}

BFbool bfGetDebugEnabled() {
#ifdef BF_DEBUG
	return BF_DEBUG;
#else
	return false;
#endif
}
BFbool bfGetCudaEnabled() {
#ifdef BF_CUDA_ENABLED
	return BF_CUDA_ENABLED;
#else
	return false;
#endif
}
