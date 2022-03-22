# Copyright (c) 2019-2021, The Bifrost Authors. All rights reserved.
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
import time
import ctypes
import threading
import bifrost as bf
from bifrost.ring import Ring
from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_writer import HeaderInfo, UDPTransmit
from bifrost.packet_capture import PacketCaptureCallback, UDPCapture
from bifrost.quantize import quantize
import numpy as np


class TBNReader(object):
    def __init__(self, sock, ring):
        self.sock = sock
        self.ring = ring
    def callback(self, seq0, time_tag, decim, chan0, nsrc, hdr_ptr, hdr_size_ptr):
        #print "++++++++++++++++ seq0     =", seq0
        #print "                 time_tag =", time_tag
        hdr = {'time_tag': time_tag,
               'seq0':     seq0, 
               'chan0':    chan0,
               'cfreq':    196e6 * chan0/2.**32,
               'bw':       196e6/decim,
               'nstand':   nsrc/2,
               'npol':     2,
               'complex':  True,
               'nbit':     8}
        #print "******** CFREQ:", hdr['cfreq']
        hdr_str = json.dumps(hdr)
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
        with UDPCapture("tbn", self.sock, self.ring, 32, 0, 9000, 49, 196,
                        sequence_callback=seq_callback) as capture:
            while True:
                status = capture.recv()
                if status in (1,4,5,6):
                    break
        del capture


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
        hdr_str = json.dumps(hdr)
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
        with UDPCapture("drx", self.sock, self.ring, self.nsrc, 0, 9000, 49, 49,
                        sequence_callback=seq_callback) as capture:
            while True:
                status = capture.recv()
                if status in (1,4,5,6):
                    break
        del capture


class PBeamReader(object):
    def __init__(self, sock, ring, nsrc=1):
        self.sock = sock
        self.ring = ring
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
        print("******** HDR:", hdr)
        hdr_str = json.dumps(hdr)
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
        with UDPCapture("pbeam", self.sock, self.ring, self.nsrc, 1, 9000, 240, 240,
                        sequence_callback=seq_callback) as capture:
            while True:
                status = capture.recv()
                if status in (1,4,5,6):
                    break
        del capture


class AccumulateOp(object):
    def __init__(self, ring, output, size, dtype=np.uint8):
        self.ring = ring
        self.output = output
        self.size = size*(dtype().nbytes)
        self.dtype = dtype
        
    def main(self):
        for iseq in self.ring.read(guarantee=True):
            iseq_spans = iseq.read(self.size)
            while not self.ring.writing_ended():
                for ispan in iseq_spans:
                    idata = ispan.data_view(self.dtype)
                    self.output.append(idata)


class UDPIOTest(unittest.TestCase):
    """Test simple IO for the UDP-based packet reader and writing"""
    def setUp(self):
        """Generate some dummy data to read"""
        # Generate test vector and save to file
        t = np.arange(256*4096*2)
        w = 0.2
        self.s0 = 5*np.cos(w * t, dtype='float32') \
                + 3j*np.sin(w * t, dtype='float32')
        
    def _get_tbn_data(self):
        # Setup the packet HeaderInfo
        desc = HeaderInfo()
        desc.set_tuning(int(round(74e6 / 196e6 * 2**32)))
        desc.set_gain(20)
        
        # Reorder as packets, stands, time
        data = self.s0.reshape(512,32,-1)
        data = data.transpose(2,1,0).copy()
        # Convert to ci8 for TBN
        data_q = bf.ndarray(shape=data.shape, dtype='ci8')
        quantize(data, data_q, scale=10)
        
        # Update the number of data sources and return
        desc.set_nsrc(data_q.shape[1])
        return desc, data_q
    def test_write_tbn(self):
        addr = Address('127.0.0.1', 7147)
        sock = UDPSocket()
        sock.connect(addr)
        op = UDPTransmit('tbn', sock)
        
        # Get TBN data
        desc, data = self._get_tbn_data()
        
        # Go!
        op.send(desc, 0, 1960*512, 0, 1, data)
        sock.close()
    def test_read_tbn(self):
        # Setup the ring
        ring = Ring(name="capture_tbn")
        
        # Setup the blocks
        addr = Address('127.0.0.1', 7147)
        ## Output via UDPTransmit
        osock = UDPSocket()
        osock.connect(addr)
        oop = UDPTransmit('tbn', osock)
        ## Input via UDPCapture
        isock = UDPSocket()
        isock.bind(addr)
        isock.timeout = 1.0
        iop = TBNReader(isock, ring)
        ## Data accumulation
        final = []
        aop = AccumulateOp(ring, final, 49*32*512*2)
        
        # Start the reader and accumlator threads
        reader = threading.Thread(target=iop.main)
        accumu = threading.Thread(target=aop.main)
        reader.start()
        accumu.start()
        
        # Get TBN data and send it off
        desc, data = self._get_tbn_data()
        for p in range(data.shape[0]):
            oop.send(desc, p*1960*512, 1960*512, 0, 1, data[p,...].reshape(1,32,512))
            time.sleep(1e-3)
        reader.join()
        accumu.join()
        
        # Compare
        ## Reorder to match what we sent out
        final = np.array(final, dtype=np.uint8)
        print('tbn_final:', final.shape)
        final = final.reshape(-1,512,32,2)
        final = final.transpose(0,2,1,3).copy()
        final = bf.ndarray(shape=(final.shape[0],32,512), dtype='ci8', buffer=final.ctypes.data)
        ## Reduce to match the capture block size
        data = data[:final.shape[0],...]
        for i in range(1, data.shape[0]):
            np.testing.assert_equal(final[i,...], data[i,...])
            
        # Clean up
        del oop
        isock.close()
        osock.close()
        
    def _get_drx_data(self):
        # Setup the packet HeaderInfo
        desc = HeaderInfo()
        desc.set_decimation(10)
        desc.set_tuning(int(round(74e6 / 196e6 * 2**32)))
        
        # Reorder as packets, beams, time
        data = self.s0.reshape(4096,4,-1)
        data = data.transpose(2,1,0).copy()
        # Convert to ci4 for DRX
        data_q = bf.ndarray(shape=data.shape, dtype='ci4')
        quantize(data, data_q)
        
        # Update the number of data sources and return
        desc.set_nsrc(data_q.shape[1])
        return desc, data_q
    def test_write_drx(self):
        addr = Address('127.0.0.1', 7147)
        sock = UDPSocket()
        sock.connect(addr)
        op = UDPTransmit('drx', sock)
        
        # Get TBN data
        desc, data = self._get_drx_data()
        
        # Go!
        op.send(desc, 0, 10*4096, (1<<3), 128, data)
        sock.close()
    def test_read_drx(self):
        # Setup the ring
        ring = Ring(name="capture_drx")
        
        # Setup the blocks
        addr = Address('127.0.0.1', 7147)
        ## Output via UDPTransmit
        osock = UDPSocket()
        osock.connect(addr)
        oop = UDPTransmit('drx', osock)
        ## Input via UDPCapture
        isock = UDPSocket()
        isock.bind(addr)
        isock.timeout = 1.0
        iop = DRXReader(isock, ring)
        ## Data accumulation
        final = []
        aop = AccumulateOp(ring, final, 49*4*4096*1)
        
        # Start the reader
        reader = threading.Thread(target=iop.main)
        accumu = threading.Thread(target=aop.main)
        reader.start()
        accumu.start()
        
        # Get DRX data and send it off
        desc, data = self._get_drx_data()
        for p in range(data.shape[0]):
            oop.send(desc, p*10*4096, 10*4096, (1<<3), 128, data[p,[0,1],...].reshape(1,2,4096))
            oop.send(desc, p*10*4096, 10*4096, (2<<3), 128, data[p,[2,3],...].reshape(1,2,4096))
            time.sleep(1e-3)
        reader.join()
        accumu.join()
        
        # Compare
        ## Reorder to match what we sent out
        final = np.array(final, dtype=np.uint8)
        final = final.reshape(-1,4096,4)
        final = final.transpose(0,2,1).copy()
        final = bf.ndarray(shape=(final.shape[0],4,4096), dtype='ci4', buffer=final.ctypes.data)
        ## Reduce to match the capture block size
        data = data[:final.shape[0],...]
        for i in range(1, data.shape[0]):
            np.testing.assert_equal(final[i,...], data[i,...])
            
        # Clean up
        del oop
        isock.close()
        osock.close()
    def test_write_drx_single(self):
        addr = Address('127.0.0.1', 7147)
        sock = UDPSocket()
        sock.connect(addr)
        op = UDPTransmit('drx', sock)
        
        # Get DRX data
        desc, data = self._get_drx_data()
        desc.set_nsrc(2)
        
        # Go!
        op.send(desc, 0, 10*4096, (1<<3), 128, data[:,[0,1],:].copy())
        sock.close()
    def test_read_drx_single(self):
        # Setup the ring
        ring = Ring(name="capture_drx_single")
        
        # Setup the blocks
        addr = Address('127.0.0.1', 7147)
        ## Output via UDPTransmit
        osock = UDPSocket()
        osock.connect(addr)
        oop = UDPTransmit('drx', osock)
        ## Input via UDPCapture
        isock = UDPSocket()
        isock.bind(addr)
        isock.timeout = 1.0
        iop = DRXReader(isock, ring, nsrc=2)
        ## Data accumulation
        final = []
        aop = AccumulateOp(ring, final, 49*2*4096*1)
        
        # Start the reader
        reader = threading.Thread(target=iop.main)
        accumu = threading.Thread(target=aop.main)
        reader.start()
        accumu.start()
        
        # Get DRX data and send it off
        desc, data = self._get_drx_data()
        desc.set_nsrc(2)
        for p in range(data.shape[0]):
            oop.send(desc, p*10*4096, 10*4096, (1<<3), 128, data[p,[0,1],:].reshape(1,2,4096))
            time.sleep(1e-3)
        reader.join()
        accumu.join()
        
        # Compare
        ## Reorder to match what we sent out
        final = np.array(final, dtype=np.uint8)
        final = final.reshape(-1,4096,2)
        final = final.transpose(0,2,1).copy()
        final = bf.ndarray(shape=(final.shape[0],2,4096), dtype='ci4', buffer=final.ctypes.data)
        ## Reduce to match the capture block size
        data = data[:final.shape[0],...]
        data = data[:,[0,1],:]
        for i in range(1, data.shape[0]):
            np.testing.assert_equal(final[i,...], data[i,...])
            
        # Clean up
        del oop
        isock.close()
        osock.close()
        
    def _get_pbeam_data(self):
        # Setup the packet HeaderInfo
        desc = HeaderInfo()
        desc.set_tuning(1)
        desc.set_chan0(345)
        desc.set_nchan(128)
        desc.set_decimation(24)
        
        # Reorder as packets, beam, chan/pol
        data = self.s0.reshape(128*4,1,-1)
        data = data.transpose(2,1,0)
        data = data.real.copy()
        
        # Update the number of data sources and return
        desc.set_nsrc(data.shape[1])
        return desc, data
    def test_write_pbeam(self):
        addr = Address('127.0.0.1', 7147)
        sock = UDPSocket()
        sock.connect(addr)
        op = UDPTransmit('pbeam1_128', sock)
        
        # Get PBeam data
        desc, data = self._get_pbeam_data()
        
        # Go!
        op.send(desc, 0, 24, 0, 1, data)
        sock.close()
    def test_read_pbeam(self):
        # Setup the ring
        ring = Ring(name="capture_pbeam")
        
        # Setup the blocks
        addr = Address('127.0.0.1', 7147)
        ## Output via UDPTransmit
        osock = UDPSocket()
        osock.connect(addr)
        oop = UDPTransmit('pbeam1_128', osock)
        ## Input via UDPCapture
        isock = UDPSocket()
        isock.bind(addr)
        isock.timeout = 1.0
        iop = PBeamReader(isock, ring, nsrc=1)
        ## Data accumulation
        final = []
        aop = AccumulateOp(ring, final, 240*128*4, dtype=np.float32)
        
        # Start the reader and accumlator threads
        reader = threading.Thread(target=iop.main)
        accumu = threading.Thread(target=aop.main)
        reader.start()
        accumu.start()
        
        # Get PBeam data and send it off
        desc, data = self._get_pbeam_data()
        for p in range(data.shape[0]):
            oop.send(desc, p*24, 24, 0, 1, data[p,...].reshape(1,1,128*4))
            time.sleep(1e-3)
        reader.join()
        accumu.join()
        
        # Compare
        ## Reorder to match what we sent out
        final = np.array(final, dtype=np.float32)
        print("final:", final.shape)
        final = final.reshape(-1,128*4,1)
        final = final.transpose(0,2,1).copy()
        ## Reduce to match the capture block size
        data = data[:(final.shape[0]//240-1)*240,...]
        for i in range(1, data.shape[0]):
            np.testing.assert_equal(final[i,...], data[i,...])
            
        # Clean up
        del oop
        isock.close()
        osock.close()
        
    def test_write_multicast(self):
        addr = Address('224.0.0.101', 7147)
        sock = UDPSocket()
        sock.connect(addr)
        op = UDPTransmit('tbn', sock)
        
        # Get TBN data
        desc, data = self._get_tbn_data()
        
        # Go!
        op.send(desc, 0, 1960*512, 0, 1, data)
        sock.close()
    def test_read_multicast(self):
        # Setup the ring
        ring = Ring(name="capture_multi")
        
        # Setup the blocks
        addr = Address('224.0.0.101', 7147)
        ## Output via UDPTransmit
        osock = UDPSocket()
        osock.connect(addr)
        oop = UDPTransmit('tbn', osock)
        ## Input via UDPCapture
        isock = UDPSocket()
        isock.bind(addr)
        isock.timeout = 1.0
        iop = TBNReader(isock, ring)
        # Data accumulation
        final = []
        aop = AccumulateOp(ring, final, 49*32*512*2)
        
        # Start the reader and accumlator threads
        reader = threading.Thread(target=iop.main)
        accumu = threading.Thread(target=aop.main)
        reader.start()
        accumu.start()
        
        # Get TBN data and send it off
        desc, data = self._get_tbn_data()
        for p in range(data.shape[0]):
            oop.send(desc, p*1960*512, 1960*512, 0, 1, data[p,...].reshape(1,32,512))
            time.sleep(1e-3)
        reader.join()
        accumu.join()
        
        # Compare
        ## Reorder to match what we sent out
        final = np.array(final, dtype=np.uint8)
        final = final.reshape(-1,512,32,2)
        final = final.transpose(0,2,1,3).copy()
        final = bf.ndarray(shape=(final.shape[0],32,512), dtype='ci8', buffer=final.ctypes.data)
        ## Reduce to match the capture block size
        data = data[:final.shape[0],...]
        for i in range(1, data.shape[0]):
            np.testing.assert_equal(final[i,...], data[i,...])
            
        # Clean up
        del oop
        isock.close()
        osock.close()
