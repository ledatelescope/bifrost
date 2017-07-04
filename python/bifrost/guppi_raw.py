
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

"""
GUPPI Raw format
Headers:
  Records are 80 chars, padded with spaces
  Keywords are truncated/padded with spaces to 8 chars
  "keyword1= <value>"
  String values are enclosed in single-quotes
  Numerical values typically not enclosed in quotes, but sometimes may be
  Final record is always 'END' + ' '*77
Header keywords:
  DIRECTIO: If present and non-zero, headers are padded to a 512-byte boundary
  NBITS:    No. bits per real component (e.g., 4 => 4+4-bit complex values);
              typical values: 8,4,2 (particularly 8)
  BLOCSIZE: No. bytes per binary data block
  OBSNCHAN (or NCHAN?)
  NPOL:     Single-pol if 1 else dual-pol
  OBSFREQ:  Centre freq of data
  OBSBW:    Bandwidth of data (may be negative to indicate high->low channel
              ordering)
  BACKEND:  'GUPPI' for guppi/BL data
  [CHAN_BW]

NTIME = BLOCSIZE * 8 / (2 * NPOL * NCHAN * NBITS)

Binary data:
  [chan][time][pol][complex]

"""

import numpy as np

def read_header(f):
    RECORD_LEN = 80
    DIRECTIO_ALIGN_NBYTE = 512
    buf = bytearray(RECORD_LEN)
    hdr = {}
    while True:
        record = f.read(RECORD_LEN)
        if len(record) < RECORD_LEN:
            raise IOError("EOF reached in middle of header")
        if record.startswith('END'):
            break
        key, val = record.split('=', 1)
        key, val = key.strip(), val.strip()
        if key in hdr:
            raise KeyError("Duplicate header key:", key)
        try: val = int(val)
        except ValueError:
            try: val = float(val)
            except ValueError:
                if val[0] not in set(["'", '"']):
                    raise ValueError("Invalid header value:", val)
                val = val[1:-1]    # Remove quotes
                val = val.rstrip() # Remove padding within string
        hdr[key] = val
    if 'DIRECTIO' in hdr:
        # Advance to alignment boundary
        # Note: We avoid using seek() so that we can support Unix pipes
        f.read(DIRECTIO_ALIGN_NBYTE - f.tell() % DIRECTIO_ALIGN_NBYTE)
    if 'NPOL' in hdr:
        # WAR for files with NPOL=4, which includes the complex components
        hdr['NPOL'] = 1 if hdr['NPOL'] == 1 else 2
    if 'NTIME' not in hdr:
        # Compute and add NTIME parameter
        hdr['NTIME'] = hdr['BLOCSIZE'] * 8 / (hdr['OBSNCHAN'] * hdr['NPOL'] *
                                              2 * hdr['NBITS'])
    return hdr

# def read_data(f, hdr):
#    assert(hdr['NBITS'] == 8)
#    count = hdr['BLOCSIZE']
#    shape = (hdr['OBSNCHAN'], hdr['NTIME'], hdr['NPOL'])
#    data = np.fromfile(f, dtype=np.int8, count=count)
#    data = data.astype(np.float32).view(np.complex64)
#    data = data.reshape(shape)
#    return data
