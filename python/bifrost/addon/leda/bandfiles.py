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

import os, sys

sys.path.append('..')
import make_header

disk_names = [ "ledastorage", "longterm", "offsite", "data" ]
ledaovro_names = [ "ledaovro1", "ledaovro2", "ledaovro3", "ledaovro4", "ledaovro5", "ledaovro6", "ledaovro7", "ledaovro8", "ledaovro9", "ledaovro10", "ledaovro11","ledaovro12" ]
data_names = [ "data1", "data2" ]
onetwo_names = [ "one", "two" ]
DADA_HEADER_SIZE = 4096
frequency_vals = [ 31.296, 33.912, 36.528, 39.144, 41.76, 44.376, 46.992, 49.608, 52.224, 54.84, 57.456, 60.072, 62.688, 65.304, 67.92, 70.536, 73.152, 75.768, 78.384, 81.0, 83.616, 86.232]


def is_integer(x): return int(x) == x

def extract_obs_offset_from_name(fname): 
  return int(os.path.basename(fname)[20:36])

def extract_obs_offset_in_file(fname):
  f = open(fname, 'rb')
  headerstr = f.read(DADA_HEADER_SIZE)
  f.close()
  if len(headerstr) < DADA_HEADER_SIZE: return "UNKNOWN"
  for line in headerstr.split('\n'):
    key, value = line.split()
    if key == "OBS_OFFSET": return int(value)

  return "UNKNOWN"

# This does work on leda_dbbeam3 files but the n_scans will be in error
class FileInfo(object):

  def __init__(self, fname):
    hedr = make_header.make_header(fname,write=False,warn=False)
    if hedr["TIME_OFFSET"] == "UNKNOWN" or hedr["N_SCANS"] == "UNKNOWN" or hedr["INT_TIME"] == "UNKNOWN" or hedr["LST"] == "UNKNOWN" or hedr["FREQCENT"] == "UNKNOWN": 
      lst = freq = t_offset = n_scans = int_time = utc_date = utc_time = 0
    else:
      t_offset = int(hedr["TIME_OFFSET"])    
      n_scans = int(hedr["N_SCANS"])
      int_time = int(hedr["INT_TIME"])
      lst = float(hedr["LST"])
      freq = float(hedr["FREQCENT"])

    self.name = fname
    self.start_time = t_offset
    self.end_time = t_offset+int(n_scans)*int_time
    self.scans = n_scans
    self.utc_date = hedr["DATE"]
    self.utc_time = hedr["TIME"]
    self.lst = lst
    self.freq = freq
    self.source = hedr["SOURCE"]
    self.mode = hedr["MODE"]
    self.size = os.path.getsize(fname)

    """
    obs1 = extract_obs_offset_from_name(fname)
    obs2 = extract_obs_offset_in_file(fname)
    if obs1 != obs2 and obs1 != "UNKNOWN" and obs2 !="UNKNOWN":
      print "Consistency Error", fname, ": OBS_OFFSET in file doesn't match the offset in the name"
    """

    if hedr["SOURCE"] == "LEDA_TEST":
      if not is_integer(n_scans): print "CONSISTENCY ERROR, ", fname, "scan:",n_scans, "is not integer"
      if not is_integer((self.end_time-self.start_time)/9.0): print "CONSISTENCY ERROR", fname, ": not 9 sec dump in file"

# Gather the file info for all files that have different frequency but the same observation time.
class BandFiles(object):

  def __init__(self, basename):

    self.basename = basename
    self.files = []
    self.start_time_present = []
    self.scans_present = []
    self.freq_present = []
    self.frequency_adjusted1 = False
    self.frequency_adjusted2 = False

    if basename[-5:] == ".dada":                # Just a single file
      if os.access(basename,os.R_OK): self.files.append(FileInfo(basename))
      else: "Error:", basename, "does not exist or is not readable"
      print basename, FileInfo(basename), self.files
    else:

      # Look for files in all the standard locations

      # build and try different names in different nfs directories
      for disk in disk_names:
        for ledaovro in ledaovro_names:
          for data in data_names:
            for onetwo in onetwo_names:
              file_name  = "/nfs/"+disk+"/"+ledaovro+"/"+data+"/"+onetwo+"/"+basename
              if os.access(file_name,os.R_OK): 
                self.files.append(FileInfo(file_name))

      # build and try different names in different local directories
      for data in data_names:
        for onetwo in onetwo_names:
          file_name  = "/"+data+"/"+onetwo+"/"+basename
          if os.access(file_name,os.R_OK):
            self.files.append(FileInfo(file_name))

    if len(self.files) == 0: return

    self.files = sorted(self.files, key=lambda fl: fl.freq)

    # Gather 
    for f in self.files:
      if f.scans not in self.scans_present: self.scans_present.append(f.scans)
      if f.freq not in self.freq_present: self.freq_present.append(f.freq)
      if f.start_time not in self.start_time_present: self.start_time_present.append(f.start_time)

    if len(self.start_time_present) > 1:
      print "Error: Files with same timestamp in their name have different internal time. Basename:",basename
      #sys.exit(1)

    self.start_time = self.start_time_present[0]
    self.end_time = self.files[0].end_time
    self.lst = self.files[0].lst
    self.utc_date = self.files[0].utc_date
    self.utc_time = self.files[0].utc_time
    self.obs_offset = extract_obs_offset_in_file(basename)

    self.ok = ( len(self.scans_present) == 1 and is_integer(self.scans_present[0]) and int(self.scans_present[0]) > 0 and self.all_frequencies_present() )

  def find_close_frequency(self, num):
    for fr in frequency_vals:
      if abs(num-fr) < 0.0000001: return True
    return False

  def all_frequencies_present(self):
    num_present = 0
    for f in self.files:
      if self.find_close_frequency(f.freq): num_present += 1

    if num_present == 0:    # Trying adjusting
      for f in self.files:
        if self.find_close_frequency(f.freq+5.244-0.012): num_present += 1
      if num_present > 0:
        self.frequency_adjusted1 = True
        for f in self.files: 
          f.freq += 5.244-0.012
      else:
        for f in self.files:
          if self.find_close_frequency(f.freq-0.012): num_present += 1
          if num_present > 0:
            self.frequency_adjusted2 = True
          for f in self.files: 
            f.freq -= 0.012

    return (num_present == 22)

  def report(self):
    frequencies = []
    scans = []
    for f in self.files:
      if f.freq not in frequencies: frequencies.append(f.freq)
      if f.scans not in scans: scans.append(f.scans)
    return frequencies, scans, self.frequency_adjusted1, self.frequency_adjusted2

  def contains_file(self, file_name):
    for f in self.files:
      if f.file_name == file_name: return True
    return False

  def has_freq(self, frequencies):
    has = []
    for f in self.files:
      if f.freq in frequencies: has.append(f.freq)
    return has

