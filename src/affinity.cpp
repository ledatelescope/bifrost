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

#include <bifrost/affinity.h>
#include "assert.hpp"

#include <omp.h>

#include <pthread.h>
//#include <sched.h>
#include <unistd.h>
#include <errno.h>

// Note: Pass core_id = -1 to unbind
BFstatus bfAffinitySetCore(int core) {
#if defined __linux__ && __linux__
	// Check for valid core
	int ncore = sysconf(_SC_NPROCESSORS_ONLN);
	BF_ASSERT(core >= -1 && core < ncore, BF_STATUS_INVALID_ARGUMENT);
	// Create core mask
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	if( core >= 0 ) {
		// Set specified core
		CPU_SET(core, &cpuset);
	}
	else {
		// Set all cores (i.e., 'un-bind')
		for( int c=0; c<ncore; ++c ) {
			CPU_SET(c, &cpuset);
		}
	}
	// Apply to current thread
	pthread_t tid = pthread_self();
	// Set affinity (note: non-portable)
	int ret = pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
	//int ret = sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);
	if( ret == 0 ) {
		return BF_STATUS_SUCCESS;
	}
	else {
		return BF_STATUS_INVALID_ARGUMENT;
	}
#else
#warning CPU core binding/affinity not supported on this OS
	return BF_STATUS_UNSUPPORTED;
#endif
}
BFstatus bfAffinityGetCore(int* core) {
	BF_ASSERT(core, BF_STATUS_INVALID_POINTER);
	pthread_t tid = pthread_self();
	cpu_set_t cpuset;
	BF_ASSERT(!pthread_getaffinity_np(tid, sizeof(cpu_set_t), &cpuset),
	          BF_STATUS_INTERNAL_ERROR);
	if( CPU_COUNT(&cpuset) > 1 ) {
		// Return -1 if more than one core is set
		// TODO: Should really check if all cores are set, otherwise fail
		*core = -1;
		return BF_STATUS_SUCCESS;
	}
	else {
		int ncore = sysconf(_SC_NPROCESSORS_ONLN);
		for( int c=0; c<ncore; ++c ) {
			if( CPU_ISSET(c, &cpuset) ) {
				*core = c;
				return BF_STATUS_SUCCESS;
			}
		}
	}
	// No cores are set! (Not sure if this is possible)
	return BF_STATUS_INVALID_STATE;
}
BFstatus bfAffinitySetOpenMPCores(BFsize     nthread,
                                  const int* thread_cores) {
	int host_core = -1;
	// TODO: Check these for errors
	bfAffinityGetCore(&host_core);
	bfAffinitySetCore(-1); // Unbind host core to unconstrain OpenMP threads
	omp_set_num_threads(nthread);
#pragma omp parallel for schedule(static, 1)
	for( BFsize t=0; t<nthread; ++t ) {
		int tid = omp_get_thread_num();
		bfAffinitySetCore(thread_cores[tid]);
	}
	return bfAffinitySetCore(host_core);
}
