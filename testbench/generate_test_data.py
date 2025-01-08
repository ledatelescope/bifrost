#!/usr/bin/env python3

# Copyright (c) 2017-2023, The Bifrost Authors. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of The Bifrost Authors nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
# generate_test_data.py

Generate test data that can be used with a testbench
"""

import os
import numpy as np

if __name__ == "__main__":
    if not os.path.exists('testdata'):
        os.mkdir('testdata')

    print("Generating sine wave dataset")
    for ii in range(32):
        # Generate test vector and save to file
        window_len = 2**16     
        n_window   = 32
        t = np.arange(window_len * n_window)
        w = 0.01
        s = np.sin((ii+1)* w * t, dtype='complex64')
        s.tofile('testdata/sin_data_%02i.bin' % ii)

    print("Generating noisy dataset")
    for ii in range(8):
        window_len = 2**18
        n_window   = 32
        noise = np.random.random(window_len * n_window).astype('complex64')
        t = np.arange(window_len * n_window, dtype='complex64')
        s0 = 1.0 * np.sin(0.01 * t, dtype='complex64')
        s1 = 2.0 * np.sin(0.07 * t, dtype='complex64')
        s2 = 4.0 * np.sin(0.12 * t, dtype='complex64')

        d = noise + s0 + s1 + s2
        d.tofile('testdata/noisy_data_%02i.bin' % ii)
        
