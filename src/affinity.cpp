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
