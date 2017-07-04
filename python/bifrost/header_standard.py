# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
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

"""@package header_standard
This file enforces a standard header for rings.

Required parameters:

(parameter type definition)
nchans int "Number of frequency channels. 1+"
nifs int "Number of separate IF channels. 1+"
nbits int "Number of bits per value. 1+"
fch1 float "Center frequency of first channel given in buffer (MHz). >0"
foff float "Bandwidth of each channel (MHz). Negative values used for when
    first channel specified has the largest frequency."
tstart float "Time stamp in MJD of first sample (seconds). >0"
tsamp float "Time interval between samples (seconds). >0"

Optional parameters (which some blocks require):

"""

# Define a header which we can check passed
# dictionaries with
# Format:
# 'parameter name':(type, minimum)
STANDARD_HEADER = {
    'nchans': (int, 1),
    'nifs': (int, 1, ),
    'nbits': (int, 1),
    'fch1': (float, 0),
    'foff': (float, None),
    'tstart': (float, 0),
    'tsamp': (float, 0)}

def enforce_header_standard(header_dict):
    """Raise an error if the header dictionary passed
        does not fit the standard specified above."""
    if type(header_dict) != dict:
        return False
    for parameter, standard in STANDARD_HEADER.items():
        if parameter not in header_dict:
            return False
        if type(header_dict[parameter]) != standard[0]:
            return False
        if standard[1] is not None and \
           header_dict[parameter] < standard[1]:
            return False

    return True
