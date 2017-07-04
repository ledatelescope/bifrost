


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
make_header.py
=======

Modified by Hugh Garsden from Danny Price's dada.py and pipeline.py

Makes header.txt files that is used by corr2uvfit and DuCT.
"""

import numpy as np
import os, sys, ephem, datetime
from dateutil import tz




class DadaReader(object):
    """ Dada file reader for raw LEDA correlator data.

    Reads the header of a dada file 

    Parameters
    ----------
    filename: str
        name of dada file to open
    n_int: int
        number of integrations to read. If None, will only read header
    inspectOnly: bool
        If inspectOnly, will only read header and will not unpack data.
    """
    DEFAULT_HEADER_SIZE = 4096

    def __init__(self, filename, warnings, file_size):
        self.filename = filename
    self.warnings = warnings
     self.file_size = file_size    # Externally supplied
    #print filename, warnings, file_size
    self.generate_info()

    def generate_info(self):
        """ Parse dada header and form useful quantities. Calculate everything that can be calculated 
        based on what's in the header. For the rest, call them UNKNOWN. """


        f = open(self.filename, 'rb')
        headerstr = f.read(self.DEFAULT_HEADER_SIZE)
    f.close()

        header = {}
        for line in headerstr.split('\n'):
            try:
                key, value = line.split()
            except ValueError:
                break
            key = key.strip()
            value = value.strip()
            header[key] = value

    self.source = header["SOURCE"]
        self.mode = header['MODE']    
    if "UTC_START" in header: self.datestamp = header['UTC_START']
        else: self.datestamp = "UNKNOWN"
        if "CFREQ" in header: self.c_freq_mhz = float(header['CFREQ'])
        else: self.c_freq_mhz = "UNKNOWN"
        if "BW" in header: self.bandwidth_mhz  = float(header['BW'])
        else: self.bandwidth_mhz = "UNKNOWN"
        if "NCHAN" in header: self.n_chans = int(header["NCHAN"])
    else: self.n_chans = "UNKNOWN"
    if "DATA_ORDER" in header: self.data_order = header["DATA_ORDER"]
    else: self.data_order = "UNKNOWN"

    have_size = True    # If we can settle on a file size for the zipped files.

        # Calculate number of integrations within this file
        # File may not be complete, hence file_size_dsk is read too.
    # However this is now complicated by zipping files. I am
    # trying to be clever to figure the size. - HG

        if self.filename[-8:] == ".dadazip":    # Will not unzip to get actual size. Must be specified somehow.
      if self.file_size:            # We are given the complete file size which overrides everything else.
        data_size_dsk = int(self.file_size)-self.DEFAULT_HEADER_SIZE
        data_size_hdr = data_size_dsk
      elif "FILE_SIZE" in header:        # Hope that this is right
        data_size_dsk = int(header["FILE_SIZE"])    # these data sizes don't include header
        data_size_hdr = data_size_dsk
      else:                    # Failure
        if self.warnings: print "WARNING: File is zipped and FILE_SIZE is not in header and file_size not supplied. "
        have_size = False
        data_size_hdr = data_size_dsk = 0
    else:                     # File not zipped. Can get true complete file size
      data_size_dsk = os.path.getsize(self.filename)-self.DEFAULT_HEADER_SIZE
      if "FILE_SIZE" in header: data_size_hdr = int(header["FILE_SIZE"])
      else: data_size_hdr = data_size_dsk

        if data_size_hdr != data_size_dsk:
      if self.warnings: print "WARNING: Data size in file doesn't match actual size. Using actual size."

    data_size = data_size_dsk        # Settle on this as the size of the data

    self.file_size = data_size+self.DEFAULT_HEADER_SIZE

    # Try to be clever and generate values that can be generated, while leaving 
     # undefined values as UNKNOWN.
        if "BYTES_PER_AVG" in header:
          bpa = int(header["BYTES_PER_AVG"])

        if "BYTES_PER_AVG" in header and have_size:
       if data_size % bpa != 0:
        if self.warnings: print "WARNING: BYTES_PER_AVG does not result in an integral number of scans"
        if "DATA_ORDER" in header and self.data_order == 'TIME_SUBSET_CHAN_TRIANGULAR_POL_POL_COMPLEX':
          if self.warnings: 
        print 'DATA_ORDER is TIME_SUBSET_CHAN_TRIANGULAR_POL_POL_COMPLEX, resetting BYTES_PER_AVG to',(109*32896*2*2+9*109*1270*2*2)*8,"(fixed)"
          bpa = (109*32896*2*2+9*109*1270*2*2)*8
          if data_size % bpa != 0 and self.warnings:
            print "WARNING: BYTES_PER_AVG still doesn't give integral number of scans"

          self.n_int = float(data_size) / bpa

    else: self.n_int = "UNKNOWN"

    if "TSAMP" in header and "NAVG" in header:
          # Calculate integration time per accumulation
          tsamp      = float(header["TSAMP"]) * 1e-6   # Sampling time per channel, in microseconds
          navg       = int(header["NAVG"])             # Number of averages per integration
          int_tim    = tsamp * navg                    # Integration time is tsamp * navg
          self.t_int = int_tim

      if "OBS_OFFSET" in header and "BYTES_PER_AVG" in header:
            # Calculate the time offset since the observation started
            byte_offset = int(header["OBS_OFFSET"])
            num_int_since_obs_start = byte_offset / bpa  
            time_offset_since_obs_start = num_int_since_obs_start * int_tim
            self.t_offset = time_offset_since_obs_start

      else: self.t_offset = "UNKNOWN"    

    else:
          self.t_int = "UNKNOWN"
          self.t_offset = "UNKNOWN"



class DadaTimes(object):
  """
    Handle the generation of true times and RA/DEC for the observation in the DADA file.
    Use pyephem for the tricky stuff. Includes the new calculation of RA/DEC in terms
    of long/lat rather than just using long/lat.
  """

  def time_at_timezone(self, dt, zone):
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz(zone)

    # Tell the datetime object that it's in UTC time zone since 
    # datetime objects are 'naive' by default
    dt = dt.replace(tzinfo=from_zone)

    # Convert time zone
    return dt.astimezone(to_zone)

  def __init__(self, header):

    ovro = ephem.Observer()
    (ovro.lat, ovro.lon, ovro.elev) = ('37.23978', '-118.281667', 1184.120)

    if header.datestamp == "UNKNOWN" or header.t_offset == "UNKNOWN":
      self.lst = "UNKNOWN"
      self.date_str = "UNKNOWN"
      self.time_str = "UNKNOWN"
      self.localtime_str = "UNKNOWN"
      self.lst_str = "UNKNOWN"
      self.dec_str = "UNKNOWN"
      return


    # Calculate times including LST
    dt = datetime.datetime.strptime(header.datestamp, "%Y-%m-%d-%H:%M:%S")+datetime.timedelta(seconds=header.t_offset)
    ovro.date = dt
    self.lst = ovro.sidereal_time()
    localt = self.time_at_timezone(dt, "America/Los_Angeles")
    self.date_str = "%04d%02d%02d"%(dt.year,dt.month,dt.day)
    self.time_str = "%02d%02d%02d"%(dt.hour,dt.minute,dt.second)
    self.localtime_str = "%02d%02d%02d"%(localt.hour,localt.minute,localt.second)
    ra, dec = ovro.radec_of(0, np.pi/2)
    self.lst_str = str(float(ra) / 2 / np.pi * 24)
    self.dec_str = str(float(repr(dec))*180/np.pi)
    #print ("UTC START:   %s"%dada_file.datestamp)
    #print ("TIME OFFSET: %s"%datetime.timedelta(seconds=dada_file.t_offset))
    #print ("NEW START:   (%s, %s)"%(date_str, time_str))


def make_header(filename, write=True, warn=True, size=None):
  """
  Create useful/necessary information about an observation. Used by other programs
  like corr2uvfits and DuCT.

  filename: DADA file, can be zipped
  warn: print warnings
  write: write a header.txt files
  size: specify a true file size in case of zipped file
  """

  # Get information from the DADA file
  dada_file = DadaReader(filename, warn, size)
  dada_times = DadaTimes(dada_file)


  # Fill and either dump or return header. Slight differences depending on which.
  header_params = {
    'N_CHANS'    : dada_file.n_chans,
    'N_SCANS'    : dada_file.n_int,
    'INT_TIME'   : dada_file.t_int,
    'FREQCENT'   : dada_file.c_freq_mhz,
    'BANDWIDTH'  : dada_file.bandwidth_mhz,
    'RA_HRS'     : dada_times.lst_str,
    'DEC_DEGS'   : dada_times.dec_str,
    'DATE'       : dada_times.date_str,
    'TIME'       : dada_times.time_str,
        'LOCALTIME'  : dada_times.localtime_str,
    'LST'         : dada_times.lst_str,
    'DATA_ORDER' : dada_file.data_order,
    'FILE_SIZE'  : dada_file.file_size,
    'MODE'       : dada_file.mode,
    'TIME_OFFSET': dada_file.t_offset,
    'SOURCE'     : dada_file.source
  }

  if header_params["N_SCANS"] == "UNKNOWN": n_scans = "UNKNOWN"
  else: n_scans = str(int(header_params['N_SCANS']))
  if write:    # This format is used by corr2uvfits and DuCT for transforming a DADA file.
    output = open("header.txt","w")
    output.write("# Generated by make_header.py\n\n")
    output.write("FIELDNAME Zenith\n")
    output.write("N_SCANS   "+n_scans+"\n")
    output.write("N_INPUTS  512\n")
    output.write("N_CHANS   "+str(header_params['N_CHANS'])+"      # number of channels in spectrum\n")
    output.write("CORRTYPE  B             # correlation type to use. 'C'(cross), 'B'(both), or 'A'(auto)\n")
    output.write("INT_TIME  "+str(header_params['INT_TIME'])+"    # integration time of scan in seconds\n")
    output.write("FREQCENT  "+str(header_params['FREQCENT'])+"    # observing center freq in MHz\n")
    output.write("BANDWIDTH "+str(header_params['BANDWIDTH'])+"   # total bandwidth in MHz\n")
    output.write("# To phase to the zenith, these must be the HA, RA and Dec of the zenith.\n")
    output.write("HA_HRS    0.000000      # the RA of the desired phase centre (hours)\n")
    output.write("RA_HRS    "+header_params['RA_HRS']+"      # the RA of the desired phase centre (hours)\n")
    output.write("DEC_DEGS  "+str(header_params['DEC_DEGS'])+"          # the DEC of the desired phase centre (degs)\n")
    output.write("DATE      "+header_params['DATE']+"        # YYYYMMDD\n")
    output.write("TIME      "+header_params['TIME']+"        # HHMMSS\n")
    output.write("LOCALTIME "+str(dada_times.localtime_str)+"\n")
    output.write("LST   "+str(dada_times.lst)+"\n")
    output.write("INVERT_FREQ 0           # 1 if the freq decreases with channel number\n")
    output.write("CONJUGATE   1           # conjugate the raw data to fix sign convention problem if necessary\n")
    output.write("GEOM_CORRECT    0\n")
    output.close()

  return header_params        # If this function is called from other scripts (e.g. plot scripts) it can supply useful information

if __name__ == "__main__":
  if len(sys.argv) == 2: make_header(sys.argv[1])
  elif len(sys.argv) == 3: make_header(sys.argv[1],size=sys.argv[2])
  else:
    print "Expecting file name and optionally file size"

