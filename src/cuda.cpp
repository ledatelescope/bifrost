/*
 * Copyright (c) 2016-2022, The Bifrost Authors. All rights reserved.
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


#include <hip/hip_runtime.h>
#include <bifrost/bfcuda.h>
#include "cuda.hpp"
#include "assert.hpp"

#if BF_CUDA_ENABLED
thread_local hipStream_t g_cuda_stream = hipStreamPerThread;
#endif

BFstatus bfStreamGet(void* stream) {
	BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
#if BF_CUDA_ENABLED
	*(hipStream_t*)stream = g_cuda_stream;
#else
	BF_FAIL("Built without CUDA support (bfStreamGet)", BF_STATUS_INVALID_STATE);
#endif
	return BF_STATUS_SUCCESS;
}
BFstatus bfStreamSet(void const* stream) {
	BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
#if BF_CUDA_ENABLED
	g_cuda_stream = *(hipStream_t*)stream;
#endif
	return BF_STATUS_SUCCESS;
}
BFstatus bfDeviceGet(int* device) {
	BF_ASSERT(device, BF_STATUS_INVALID_POINTER);
#if BF_CUDA_ENABLED
	BF_CHECK_HIP(hipGetDevice(device), BF_STATUS_DEVICE_ERROR);
#else
	*device = -1;
#endif
	return BF_STATUS_SUCCESS;
}
BFstatus bfDeviceSet(int device) {
#if BF_CUDA_ENABLED
	BF_CHECK_HIP(hipSetDevice(device), BF_STATUS_DEVICE_ERROR);
#endif
	return BF_STATUS_SUCCESS;
}
BFstatus bfDeviceSetById(const char* pci_bus_id) {
#if BF_CUDA_ENABLED
	int device;
	BF_CHECK_HIP(hipDeviceGetByPCIBusId(&device, pci_bus_id),
	              BF_STATUS_DEVICE_ERROR);
	return bfDeviceSet(device);
#else
	return BF_STATUS_SUCCESS;
#endif
}
BFstatus bfStreamSynchronize() {
#if BF_CUDA_ENABLED
	BF_CHECK_HIP(hipStreamSynchronize(g_cuda_stream),
	              BF_STATUS_DEVICE_ERROR);
#endif
	return BF_STATUS_SUCCESS;
}
BFstatus bfDevicesSetNoSpinCPU() {
#if BF_CUDA_ENABLED
	int old_device;
	BF_CHECK_HIP(hipGetDevice(&old_device), BF_STATUS_DEVICE_ERROR);
	int ndevices;
	BF_CHECK_HIP(hipGetDeviceCount(&ndevices), BF_STATUS_DEVICE_ERROR);
	for( int d=0; d<ndevices; ++d ) {
		BF_CHECK_HIP(hipSetDevice(d), BF_STATUS_DEVICE_ERROR);
		BF_CHECK_HIP(hipSetDeviceFlags(hipDeviceScheduleBlockingSync),
		              BF_STATUS_DEVICE_ERROR);
	}
	BF_CHECK_HIP(hipSetDevice(old_device), BF_STATUS_DEVICE_ERROR);
#endif
	return BF_STATUS_SUCCESS;
}
