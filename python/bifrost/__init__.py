
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

"""
Bifrost pipeline processing library
"""

__version__    = "0.5"
__author__     = "Ben Barsdell"
__copyright__  = "Copyright 2016, Ben Barsdell"
__credits__    = ["Ben Barsdell"]
__license__    = "Apache v2"
__maintainer__ = "Ben Barsdell"
__email__      = "benbarsdell@gmail.com"
__status__     = "Development"

# TODO: Not sure what's best here
#import bifrost
#__all__ = ['bifrost']
##from bifrost import Ring
#from ring import Ring
#from GPUArray import GPUArray
#from bifrost import bfMemcpy2D, bfTranspose
#from udp import Address, UDPSocket, CHIPSCapture

#from affinity import get_core, set_core

#import affinity
#__all__ = ['affinity']
import core, memory, affinity
#from bifrost import bf as bf
