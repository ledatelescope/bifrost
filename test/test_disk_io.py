# Copyright (c) 2019-2024, The Bifrost Authors. All rights reserved.
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

import unittest
import os
import json
import ctypes
import threading
import bifrost as bf
from bifrost.ring import Ring
from bifrost.packet_writer import HeaderInfo, DiskWriter
from bifrost.packet_capture import PacketCaptureCallback, DiskReader
from bifrost.quantize import quantize
from bifrost.pipeline import SourceBlock, SinkBlock
import datetime
import numpy as np

class AccumulateOp(object):
    def __init__(self, ring, output_timetags, output_data, size, dtype=np.uint8):
        self.ring = ring
        self.output_timetags = output_timetags
        self.output_data = output_data
        self.size = size*(dtype().nbytes)
        self.dtype = dtype
        
    def main(self):
        for iseq in self.ring.read(guarantee=True):
            self.output_timetags.append(iseq.time_tag)
            self.output_data.append([])
            
            iseq_spans = iseq.read(self.size)
            while not self.ring.writing_ended():
                for ispan in iseq_spans:
                    idata = ispan.data_view(self.dtype)
                    self.output_data[-1].append(idata.copy())

class BaseDiskIOTest(object):
    class BaseDiskIOTestCase(unittest.TestCase):
        def setUp(self):
            """Generate some dummy data to read"""
            # Generate test vector and save to file
            t = np.arange(256*4096*2)
            w = 0.2
            self.s0 = 5*np.cos(w * t, dtype='float32') \
                    + 3j*np.sin(w * t, dtype='float32')
            # Filename cache so we can cleanup later
            self._cache = []
        def _open(self, filename, mode):
            fh = open(filename, mode)
            if filename not in self._cache:
                self._cache.append(filename)
            return fh
        def tearDown(self):
            for filename in self._cache:
                os.unlink(filename)


class TBNReader(object):
    def __init__(self, sock, ring, nsrc=32):
        self.sock = sock
        self.ring = ring
        self.nsrc = nsrc
    def callback(self, seq0, time_tag, decim, chan0, nsrc, hdr_ptr, hdr_size_ptr):
        hdr = {'time_tag': time_tag,
               'seq0':     seq0, 
               'chan0':    chan0,
               'cfreq':    196e6 * chan0/2.**32,
               'bw':       196e6/decim,
               'nstand':   nsrc/2,
               'npol':     2,
               'complex':  True,
               'nbit':     8}
        try:
            hdr_str = json.dumps(hdr).encode()
        except AttributeError:
            # Python2 catch
            pass
        # TODO: Can't pad with NULL because returned as C-string
        #hdr_str = json.dumps(hdr).ljust(4096, '\0')
        #hdr_str = json.dumps(hdr).ljust(4096, ' ')
        header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0]      = ctypes.cast(header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        return 0
    def main(self):
        seq_callback = PacketCaptureCallback()
        seq_callback.set_tbn(self.callback)
        with DiskReader("tbn", self.sock, self.ring, self.nsrc, 0, 16, 128,
                        sequence_callback=seq_callback) as capture:
            while True:
                status = capture.recv()
                if status in (1,4,5,6):
                    break
        del capture
        
class TBNDiskIOTest(BaseDiskIOTest.BaseDiskIOTestCase):
    """Test simple IO for the disk-based TBN packet reader and writing"""
    def _get_data(self):
        # Setup the packet HeaderInfo
        hdr_desc = HeaderInfo()
        hdr_desc.set_tuning(int(round(74e6 / 196e6 * 2**32)))
        hdr_desc.set_gain(20)
        
        # Reorder as packets, stands, time
        data = self.s0.reshape(512,32,-1)
        data = data.transpose(2,1,0).copy()
        # Convert to ci8 for TBN
        data_q = bf.ndarray(shape=data.shape, dtype='ci8')
        quantize(data, data_q, scale=10)
        
        # Update the number of data sources and return
        hdr_desc.set_nsrc(data_q.shape[1])
        return 1, hdr_desc, data_q
    def test_write(self):
        fh = self._open('test_tbn.dat', 'wb')
        oop = DiskWriter('tbn', fh)
        
        # Get TBN data
        timetag0, hdr_desc, data = self._get_data()
        
        # Go!
        oop.send(hdr_desc, timetag0, 1960*512, 0, 1, data)
        fh.close()
        
        self.assertEqual(os.path.getsize('test_tbn.dat'), \
                        1048*data.shape[0]*data.shape[1])
    def test_read(self):
        # Write
        fh = self._open('test_tbn.dat', 'wb')
        oop = DiskWriter('tbn', fh)
        
        # Get TBN data
        timetag0, hdr_desc, data = self._get_data()
        
        # Go!
        oop.send(hdr_desc, timetag0, 1960*512, 0, 1, data)
        fh.close()
        
        # Read
        fh = self._open('test_tbn.dat', 'rb')
        ring = Ring(name="capture_tbn")
        iop = TBNReader(fh, ring, nsrc=32)
        ## Data accumulation
        times = []
        final = []
        aop = AccumulateOp(ring, times, final, 32*512*2)
        
        # Start the reader and accumlator threads
        reader = threading.Thread(target=iop.main)
        accumu = threading.Thread(target=aop.main)
        reader.start()
        accumu.start()
        
        # Get TBN data
        reader.join()
        accumu.join()
        
        # Compare
        for seq_timetag,seq_data in zip(times, final):
            ## Loop over sequences
            seq_data = np.array(seq_data, dtype=np.uint8)
            seq_data = seq_data.reshape(-1,512,32,2)
            seq_data = seq_data.transpose(0,2,1,3).copy()
            ## Drop the last axis (complexity) since we are going to ci8
            seq_data = bf.ndarray(shape=seq_data.shape[:-1], dtype='ci8', buffer=seq_data.ctypes.data)
            
            ## Ignore the first set of packets
            np.testing.assert_equal(seq_data[1:,...], data[1:,...])
            
        # Clean up
        del oop
        fh.close()


class DRXReader(object):
    def __init__(self, sock, ring, nsrc=4):
        self.sock = sock
        self.ring = ring
        self.nsrc = nsrc
    def callback(self, seq0, time_tag, decim, chan0, chan1, nsrc, hdr_ptr, hdr_size_ptr):
        #print "++++++++++++++++ seq0     =", seq0
        #print "                 time_tag =", time_tag
        hdr = {'time_tag': time_tag,
               'seq0':     seq0, 
               'chan0':    chan0,
               'chan1':    chan1,
               'cfreq0':   196e6 * chan0/2.**32,
               'cfreq1':   196e6 * chan1/2.**32,
               'bw':       196e6/decim,
               'nstand':   nsrc/2,
               'npol':     2,
               'complex':  True,
               'nbit':     4}
        #print "******** CFREQ:", hdr['cfreq']
        try:
            hdr_str = json.dumps(hdr).encode()
        except AttributeError:
            # Python2 catch
            pass
        # TODO: Can't pad with NULL because returned as C-string
        #hdr_str = json.dumps(hdr).ljust(4096, '\0')
        #hdr_str = json.dumps(hdr).ljust(4096, ' ')
        header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0]      = ctypes.cast(header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        return 0
    def main(self):
        seq_callback = PacketCaptureCallback()
        seq_callback.set_drx(self.callback)
        with DiskReader("drx", self.sock, self.ring, self.nsrc, 0, 16, 128,
                        sequence_callback=seq_callback) as capture:
            while True:
                status = capture.recv()
                if status in (1,4,5,6):
                    break
        del capture

class DRXDiskIOTest(BaseDiskIOTest.BaseDiskIOTestCase):
    """Test simple IO for the disk-based DRX packet reader and writing"""
    def _get_data(self):
        # Setup the packet HeaderInfo
        hdr_desc = HeaderInfo()
        hdr_desc.set_decimation(10)
        hdr_desc.set_tuning(int(round(74e6 / 196e6 * 2**32)))
        
        # Reorder as packets, beams, time
        data = self.s0.reshape(4096,4,-1)
        data = data.transpose(2,1,0).copy()
        # Convert to ci4 for DRX
        data_q = bf.ndarray(shape=data.shape, dtype='ci4')
        quantize(data, data_q)
        
        # Update the number of data sources and return
        hdr_desc.set_nsrc(data_q.shape[1])
        return 1, hdr_desc, data_q
    def test_write(self):
        fh = self._open('test_drx.dat', 'wb')
        oop = DiskWriter('drx', fh)
        
        # Get DRX data
        timetag0, hdr_desc, data = self._get_data()
        
        # Go!
        oop.send(hdr_desc, timetag0, 10*4096, (1<<3), 128, data)
        fh.close()
        
        self.assertEqual(os.path.getsize('test_drx.dat'), \
                        4128*data.shape[0]*data.shape[1])
    def test_read(self):
        # Write
        fh = self._open('test_drx.dat', 'wb')
        oop = DiskWriter('drx', fh)
        
        # Get DRX data and write it out
        timetag0, hdr_desc, data = self._get_data()
        for p in range(data.shape[0]):
            oop.send(hdr_desc, timetag0+p*10*4096, 10*4096, (1<<3), 128, data[p,[0,1],...].reshape(1,2,4096))
            oop.send(hdr_desc, timetag0+p*10*4096, 10*4096, (2<<3), 128, data[p,[2,3],...].reshape(1,2,4096))
        fh.close()
        
        # Read
        fh = self._open('test_drx.dat', 'rb')
        ring = Ring(name="capture_drx")
        iop = DRXReader(fh, ring, nsrc=4)
        ## Data accumulation
        times = []
        final = []
        aop = AccumulateOp(ring, times, final, 4*4096*1)
        
        # Start the reader
        reader = threading.Thread(target=iop.main)
        accumu = threading.Thread(target=aop.main)
        reader.start()
        accumu.start()
        
        # Get DRX data
        reader.join()
        accumu.join()
        
        # Compare
        for seq_timetag,seq_data in zip(times, final):
            ## Reorder to match what we sent out
            seq_data = np.array(seq_data, dtype=np.uint8)
            seq_data = seq_data.reshape(-1,4096,4)
            seq_data = seq_data.transpose(0,2,1).copy()
            seq_data = bf.ndarray(shape=seq_data.shape, dtype='ci4', buffer=seq_data.ctypes.data)
            
            np.testing.assert_equal(seq_data[1:,...], data[1:,...])
            
        # Clean up
        del oop
        fh.close()
    def test_write_single(self):
        fh = self._open('test_drx_single.dat', 'wb')
        oop = DiskWriter('drx', fh)
        
        # Get DRX data
        timetag0, hdr_desc, data = self._get_data()
        hdr_desc.set_nsrc(2)
        data = data[:,[0,1],:].copy()
        
        # Go!
        oop.send(hdr_desc, timetag0, 10*4096, (1<<3), 128, data)
        fh.close()
        
        self.assertEqual(os.path.getsize('test_drx_single.dat'), \
                        4128*data.shape[0]*data.shape[1])
    def test_read_single(self):
        # Write
        fh = self._open('test_drx_single.dat', 'wb')
        oop = DiskWriter('drx', fh)
        
        # Get DRX data and write it out
        timetag0, hdr_desc, data = self._get_data()
        hdr_desc.set_nsrc(2)
        data = data[:,[0,1],:].copy()
        oop.send(hdr_desc, timetag0, 10*4096, (1<<3), 128, data)
        fh.close()
        
        # Read
        fh = self._open('test_drx_single.dat', 'rb')
        ring = Ring(name="capture_drx_single")
        iop = DRXReader(fh, ring, nsrc=2)
        ## Data accumulation
        times = []
        final = []
        aop = AccumulateOp(ring, times, final, 2*4096*1)
        
        # Start the reader
        reader = threading.Thread(target=iop.main)
        accumu = threading.Thread(target=aop.main)
        reader.start()
        accumu.start()
        
        # Get DRX data
        reader.join()
        accumu.join()
        
        # Compare
        for seq_timetag,seq_data in zip(times, final):
            ## Reorder to match what we sent out
            seq_data = np.array(seq_data, dtype=np.uint8)
            seq_data = seq_data.reshape(-1,4096,2)
            seq_data = seq_data.transpose(0,2,1).copy()
            seq_data = bf.ndarray(shape=seq_data.shape, dtype='ci4', buffer=seq_data.ctypes.data)
            
            np.testing.assert_equal(seq_data[1:,...], data[1:,...])
            
        # Clean up
        del oop
        fh.close()


class PBeamReader(object):
    def __init__(self, sock, ring, nchan, nsrc=1):
        self.sock = sock
        self.ring = ring
        self.nchan = nchan
        self.nsrc = nsrc
    def callback(self, seq0, time_tag, navg, chan0, nchan, nbeam, hdr_ptr, hdr_size_ptr):
        #print "++++++++++++++++ seq0     =", seq0
        #print "                 time_tag =", time_tag
        hdr = {'time_tag': time_tag,
               'seq0':     seq0, 
               'chan0':    chan0,
               'cfreq0':   chan0*(196e6/8192),
               'bw':       nchan*(196e6/8192),
               'navg':     navg,
               'nbeam':    nbeam,
               'npol':     4,
               'complex':  False,
               'nbit':     32}
        #print("******** HDR:", hdr)
        try:
            hdr_str = json.dumps(hdr).encode()
        except AttributeError:
            # Python2 catch
            pass
        # TODO: Can't pad with NULL because returned as C-string
        #hdr_str = json.dumps(hdr).ljust(4096, '\0')
        #hdr_str = json.dumps(hdr).ljust(4096, ' ')
        header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0]      = ctypes.cast(header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        return 0
    def main(self):
        seq_callback = PacketCaptureCallback()
        seq_callback.set_pbeam(self.callback)
        with DiskReader("pbeam_%i" % self.nchan, self.sock, self.ring, self.nsrc, 1, 16, 128,
                        sequence_callback=seq_callback) as capture:
            while True:
                status = capture.recv()
                if status in (1,4,5,6):
                    break
        del capture

class PBeamDiskIOTest(BaseDiskIOTest.BaseDiskIOTestCase):
    """Test simple IO for the disk-based PBeam packet reader and writing"""
    def _get_data(self):
        # Setup the packet HeaderInfo
        hdr_desc = HeaderInfo()
        hdr_desc.set_tuning(2)
        hdr_desc.set_chan0(1034)
        hdr_desc.set_nchan(128)
        hdr_desc.set_decimation(24)
        
        # Reorder as packets, beam, chan/pol
        data = self.s0.reshape(128*4,1,-1)
        data = data.transpose(2,1,0)
        data = data.real[:1024,...].copy()
        
        # Update the number of data sources and return
        hdr_desc.set_nsrc(data.shape[1])
        return 1, hdr_desc, data
    def test_write(self):
       fh = self._open('test_pbeam.dat', 'wb')
       oop = DiskWriter('pbeam1_128', fh)
       
       # Get PBeam data
       timetag0, hdr_desc, data = self._get_data()
       
       # Go!
       oop.send(hdr_desc, timetag0, 24, 0, 1, data)
       fh.close()
       
       self.assertEqual(os.path.getsize('test_pbeam.dat'), \
                       (18+128*4*4)*data.shape[0]*data.shape[1])
    def test_read(self):
        # Write
        fh = self._open('test_pbeam.dat', 'wb')
        oop = DiskWriter('pbeam1_128', fh)
        
        # Get PBeam data
        timetag0, hdr_desc, data = self._get_data()
        
        # Go!
        oop.send(hdr_desc, timetag0, 24, 0, 1, data)
        fh.close()
        
        # Read
        fh = self._open('test_pbeam.dat', 'rb')
        ring = Ring(name="capture_pbeam")
        iop = PBeamReader(fh, ring, 128, nsrc=1)
        ## Data accumulation
        times = []
        final = []
        aop = AccumulateOp(ring, times, final, 1*128*4, dtype=np.float32)
        
        # Start the reader and accumlator threads
        reader = threading.Thread(target=iop.main)
        accumu = threading.Thread(target=aop.main)
        reader.start()
        accumu.start()
        
        # Get PBeam data
        reader.join()
        accumu.join()
        
        # Compare
        for seq_timetag,seq_data in zip(times, final):
            ## Reorder to match what we sent out
            seq_data = np.array(seq_data, dtype=np.float32)
            seq_data = seq_data.reshape(-1,128*4,1)
            seq_data = seq_data.transpose(0,2,1).copy()
            
            np.testing.assert_equal(seq_data[1:,...], data[1:,...])
            
        # Clean up
        del oop
        fh.close()


start_pipeline = datetime.datetime.now()

class SIMPLEReader(object):
    def __init__(self, sock, ring):
        self.sock = sock
        self.ring = ring
        self.nsrc = 1
    def seq_callback(self, seq0, chan0, nchan, nsrc,
                     time_tag_ptr, hdr_ptr, hdr_size_ptr):
        FS = 196.0e6
        CHAN_BW = 1e3
       #  timestamp0 = (self.utc_start - ADP_EPOCH).total_seconds()
       #  time_tag0  = timestamp0 * int(FS)
        time_tag   = int((datetime.datetime.now() - start_pipeline).total_seconds()*1e6)
        time_tag_ptr[0] = time_tag
        cfreq = 55e6
        hdr = {'time_tag': time_tag,
               'seq0':     seq0,
               'chan0':    chan0,
               'nchan':    nchan,
               'cfreq':    cfreq,
               'bw':       CHAN_BW,
               'nstand':   1,
               #'stand0':   src0*16, # TODO: Pass src0 to the callback too(?)
               'npol':     1,
               'complex':  True,
               'nbit':     16}
        hdr_str = json.dumps(hdr).encode()
        # TODO: Can't pad with NULL because returned as C-string
        #hdr_str = json.dumps(hdr).ljust(4096, '\0')
        #hdr_str = json.dumps(hdr).ljust(4096, ' ')
        self.header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0]      = ctypes.cast(self.header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        return 0
    def main(self):
        seq_callback = PacketCaptureCallback()
        seq_callback.set_simple(self.seq_callback)
        with DiskReader("simple" , self.sock, self.ring, self.nsrc, 0, 16, 128,
                        sequence_callback=seq_callback) as capture:
            while True:
                status = capture.recv()
                if status in (1,4,5,6):
                    break
        del capture

class SimpleDiskIOTest(BaseDiskIOTest.BaseDiskIOTestCase):
    """Test simple IO for the disk-based Simple packet reader and writing"""
    def _get_simple_data(self):
        hdr_desc = HeaderInfo()
        
        # Reorder as packets, stands, time
        data = self.s0.reshape(2048,1,-1)
        data = data.transpose(2,1,0).copy()
        # Convert to ci16 for simple
        data_q = bf.ndarray(shape=data.shape, dtype='ci16')
        quantize(data, data_q, scale=10)
        
        return 128, hdr_desc, data_q

    def test_write_simple(self):
        fh = self._open('test_simple.dat','wb')
        oop = DiskWriter('simple', fh)
        timetag0, hdr_desc, data = self._get_simple_data()
        oop.send(hdr_desc, timetag0, 1, 0, 1, data)
        fh.close()
        
        nelements = 2048
        byteperelem = 4
        bytehdrperelem = 8
        npackets = 1024 
        expectedsize = (2048*4 + 8)*1024
        self.assertEqual(os.path.getsize('test_simple.dat'), \
                        expectedsize)

    def test_read_simple(self):
        # Write
        fh = self._open('test_simple.dat', 'wb')
        oop = DiskWriter('simple', fh)
        
        # Get data
        timetag0, hdr_desc, data = self._get_simple_data()
        
        # Go!
        oop.send(hdr_desc, timetag0, 1, 0, 1, data)
        fh.close()
        
        # Read
        
        fh = self._open('test_simple.dat', 'rb')
        ring = Ring(name="capture_simple")
        iop = SIMPLEReader(fh, ring)
        ## Data accumulation
        times = []
        final = []
        expectedsize = 1*2048*4
        aop = AccumulateOp(ring, times, final, expectedsize, dtype=np.uint16)
        
        # Start the reader and accumlator threads
        reader = threading.Thread(target=iop.main)
        accumu = threading.Thread(target=aop.main)
        reader.start()
        accumu.start()
        
        # Get simple data
        reader.join()
        accumu.join()
        
        # Compare
        for seq_timetag,seq_data in zip(times, final):
            seq_data = np.array(seq_data, dtype=np.uint16)
            seq_data = seq_data.reshape(-1,2048,1,2)
            seq_data = seq_data.transpose(0,2,1,3).copy()
            ## Drop the last axis (complexity) since we are going to ci16
            seq_data = bf.ndarray(shape=seq_data.shape[:-1], dtype='ci16', buffer=seq_data.ctypes.data)
            
            np.testing.assert_equal(seq_data[1:,...], data[1:,...])
            
        del oop
        fh.close()
