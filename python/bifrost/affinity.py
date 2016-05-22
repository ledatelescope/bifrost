
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

from libbifrost import _bf, _check, _get, _array

def get_core():
	return _get(_bf.AffinityGetCore())
def set_core(core):
	_check(_bf.AffinitySetCore(core))
def set_openmp_cores(cores):
	# PYCLIBRARY ISSUE
	# TODO: Would be really nice to be able to directly pass
	#         a list here instead of needing to specify _array+type.
	#         Should be able to do it safely for any const* argument
	#         Note that the base type of the pointer type could be
	#           derived via a reverse lookup table.
	#           E.g., Inverse of POINTER(c_int)-->LP_c_int
	_check(_bf.AffinitySetOpenMPCores(len(cores), _array('int', cores)))
