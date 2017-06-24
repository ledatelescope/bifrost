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

"""@package blocks
This file contains blocks specific to LEDA-OVRO.
"""

import os
import bandfiles
import bifrost
import json
import numpy as np
from bifrost import block


# Read a collection of DADA files and form an array of time series data over
# many frequencies.
# TODO: Add more to the header.
# Add a list of frequencies present. This could be the full band,
# but there could be gaps. Downstream functionality has to be aware of the gaps.
# Allow specification of span size based on time interval
class DadaReadBlock(object):

  # Assemble a group of files in the time direction and the frequency direction
  # time_stamp is of the form "2016-05-24-11:04:38", or a DADA file ending in .dada
  def __init__(self, time_stamp, core=-1, gulp_nframe=4096):
    self.CHANNEL_WIDTH = 0.024
    self.SAMPLING_RATE = 41.66666666667e-6
    self.N_CHAN = 109
    self.N_BEAM = 2
    self.HEADER_SIZE = 4096
    self.OBS_OFFSET = 1255680000
    self.gulp_nframe = gulp_nframe

    self.core = core

    beamformer_scans = []
    if time_stamp[-5:] == ".dada": beamformer_scans.append(bandfiles.BandFiles(time_stamp))  # Just one file
    else:
      finish = False
      i = 0
      while not finish:         # This loop gathers files by time
        new_offset = str(i*self.OBS_OFFSET)
        file_name = time_stamp+"_"+"0"*(16-len(new_offset))+new_offset+".000000.dada"

        scan = bandfiles.BandFiles(file_name)           # The BandFiles class gathers by frequency

        finish = ( len(scan.files) == 0 )

        if not finish: beamformer_scans.append(scan)

        i += 1

    # Report what we've got
    print "Num files in time:",  len(beamformer_scans)
    print "File and number:"
    for scan in beamformer_scans:
      print os.path.basename(scan.files[0].name)+":", len(scan.files)

    self.beamformer_scans = beamformer_scans     # List of full-band time steps

  def main(self, input_rings, output_rings):
    bifrost.affinity.set_core(self.core)

    self.oring = output_rings[0]
    # Calculate some constants for sizes
    length_one_second = int(round(1/self.SAMPLING_RATE))
    ring_span_size = length_one_second*self.N_CHAN*4                    # 1 second, all the channels (109) and then 4-byte floats
    file_chunk_size = length_one_second*self.N_BEAM*self.N_CHAN*2               # 1 second, 2 beams, 109 chans, and 2 1-byte ints (real, imag)
    number_of_seconds = 120     # Change this

    ohdr = {}
    ohdr["frame_shape"] = ( self.N_CHAN, 1 )
    ohdr["nbit"] = 32
    ohdr["dtype"] = str(np.float32)
    ohdr["tstart"] = 0
    ohdr["tsamp"] = self.SAMPLING_RATE
    ohdr['foff'] = self.CHANNEL_WIDTH

    #print length_one_second, ring_span_size, file_chunk_size, number_of_chunks

    with self.oring.begin_writing() as oring:

      # Go through the files by time. 
      for scan in self.beamformer_scans:
        # Go through the frequencies
        for f in scan.files:

          print "Opening", f.name

          with open(f.name,'rb') as ifile:
            ifile.read(self.HEADER_SIZE)

            ohdr["cfreq"] = f.freq
            ohdr["fch1"] = f.freq

            self.oring.resize(ring_span_size)
            with oring.begin_sequence(f.name, header=json.dumps(ohdr)) as osequence:

              for i in range(number_of_seconds):
                  # Get a chunk of data from the file. The whole band is used, but only a chunk of time (1 second).
                  # Massage the data so it can go through the ring. That means changng the data type and flattening.
                try:
                    data = np.fromfile(ifile, count=file_chunk_size, dtype=np.int8).astype(np.float32)
                except:
                    print "Bad read. Stopping read."
                    return
                if data.size != length_one_second*self.N_BEAM*self.N_CHAN*2:
                    print "Bad data shape. Stopping read."
                    return
                data = data.reshape(length_one_second, self.N_BEAM, self.N_CHAN, 2)
                power = (data[...,0]**2 + data[...,1]**2).mean(axis=1)  # Now have time by frequency.
                # Send the data
                with osequence.reserve(ring_span_size) as wspan:
                  wspan.data[0][:] = power.view(dtype=np.uint8).ravel()
