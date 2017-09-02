
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

import unittest
import numpy as np
import bifrost as bf
from bifrost.DataType import DataType

from bifrost.blocks import *
from bifrost.pipeline import SourceBlock, SinkBlock

RTOL = 1e-4
ATOL = 1e-5

class DictReader(object):
    def __init__(self, dict_):
        self.dict = dict_
    def __enter__(self):
        return self.dict
    def __exit__(self, type, value, tb):
        pass

class CorrelateTestInputBlock(SourceBlock):
    def create_reader(self, idict):
        return DictReader(idict)
    def on_sequence(self, idict, _):
        ohdr = {
            '_tensor': {
                'dtype':  'ci8',
                'shape':  [-1, idict['nchan'], idict['nstation'], idict['npol']],
                'labels': ['time', 'freq', 'station', 'pol'],
                'scales': [(0, 1./idict['chan_bw']), (idict['cfreq'], idict['chan_bw']), None, None],
                'units':  ['s', 'Hz', None, ('X', 'Y')]
            }
        }
        self.nframe = idict['ntime']
        return [ohdr]
    def on_data(self, reader, ospans):
        ospan = ospans[0]
        odata = ospan.data
        i = np.arange(odata.shape[-2] * odata.shape[-1] * 2) % 255 - 127
        odata.view(np.int8)[...] = i.reshape((odata.shape[-2], odata.shape[-1]*2))
        nframe = min(ospan.nframe, self.nframe)
        self.nframe -= nframe
        return [nframe]

class CallbackBlock(SinkBlock):
    """Testing-only block which calls user-defined
        functions on sequence and on data"""
    def __init__(self, iring, seq_callback, data_callback, data_ref=None,
                 *args, **kwargs):
        super(CallbackBlock, self).__init__(iring, *args, **kwargs)
        self.seq_callback  = seq_callback
        self.data_callback = data_callback
        self.data_ref = data_ref
    def on_sequence(self, iseq):
        if self.seq_callback is not None:
            self.seq_callback(iseq)
    def on_data(self, ispan):
        if self.data_callback is not None:
            self.data_callback(ispan)
        if self.data_ref is not None:
            # Note: This can be used to check data from outside the pipeline,
            #         which is useful when exceptions inside blocks prevent
            #         downstream callback blocks from ever executing.
            self.data_ref['idata'] = ispan.data.copy()

class PipelineTest(unittest.TestCase):
    def setUp(self):
        # Note: This file needs to be large enough to fill the minimum-size
        #         ring buffer at least a few times over in order to properly
        #         test things.
        self.fil_file = "./data/2chan16bitNoDM.fil"
    def test_cuda_copy(self):
        def check_sequence(seq):
            pass
        def check_data(ispan):
            pass
        gulp_nframe = 101
        with bf.Pipeline() as pipeline:
            data = read_sigproc([self.fil_file], gulp_nframe)
            for _ in xrange(10):
                data = copy(data, space='cuda')
                data = copy(data, space='cuda_host')
            ref = {}
            CallbackBlock(data, check_sequence, check_data, data_ref=ref)
            pipeline.run()
            self.assertEqual(ref['idata'].dtype, 'uint16')
            self.assertEqual(ref['idata'].shape, (29, 1, 2))
    def test_fdmt(self):
        gulp_nframe = 101
        # TODO: Check handling of multiple pols (not currently supported?)
        def check_sequence(seq):
            hdr = seq.header
            tensor = hdr['_tensor']
            self.assertEqual(tensor['shape'],  [1,5,-1])
            self.assertEqual(tensor['dtype'],  'f32')
            self.assertEqual(tensor['labels'], ['pol', 'dispersion', 'time'])
            self.assertEqual(tensor['units'],  [None, 'pc cm^-3', 's'])
            self.assertEqual(hdr['cfreq_units'], 'MHz')
            self.assertEqual(hdr['cfreq'], 433.937)
        def check_data(ispan):
            # Note: nframe = gulp_nframe + max_delay
            self.assertEqual(ispan.data.shape, (1,5,ispan.nframe))
        with bf.Pipeline() as pipeline:
            data = read_sigproc([self.fil_file], gulp_nframe)
            data = copy(data, space='cuda')
            data = transpose(data, ['pol', 'freq', 'time'])
            data = fdmt(data, max_dm=30.)
            ref = {}
            CallbackBlock(data, check_sequence, check_data, data_ref=ref)
            data = transpose(data, ['time', 'pol', 'dispersion'])
            data = copy(data, space='cuda_host')
            pipeline.run()
            self.assertEqual(ref['idata'].dtype, 'float32')
            #self.assertEqual(ref['idata'].shape, (1, 5, 17))
            self.assertEqual(ref['idata'].shape, (1, 5, 24)) # TODO: Need to check this against an absolute somehow
    def test_reduce(self):
        gulp_nframe = 128
        nreduce_freq = 2
        nreduce_time = 8
        def check_sequence(seq):
            tensor = seq.header['_tensor']
            self.assertEqual(seq.header['gulp_nframe'], gulp_nframe // nreduce_time)
            self.assertEqual(tensor['shape'],  [-1,1,2 // nreduce_freq])
            self.assertEqual(tensor['dtype'],  'f32')
            self.assertEqual(tensor['labels'], ['time', 'pol', 'freq'])
            self.assertEqual(tensor['units'],  ['s', None, 'MHz'])
        def check_data(ispan):
            pass
        with bf.Pipeline() as pipeline:
            data = read_sigproc([self.fil_file], gulp_nframe)
            data = copy(data, space='cuda')
            data = bf.blocks.reduce(data, 'freq', nreduce_freq)
            data = bf.blocks.reduce(data, 'time', nreduce_time)
            CallbackBlock(data, check_sequence, check_data)
            pipeline.run()
    def test_correlate(self):
        gulp_nframe  = 100
        nreduce_time = 1000
        metadata = {
            'ntime':  10000,
            'nchan':    128,
            'nstation':  60,
            'npol':       2,
            'chan_bw': 25e3,
            'cfreq':   50e6
        }
        def check_sequence(seq):
            tensor = seq.header['_tensor']
            self.assertEqual(seq.header['gulp_nframe'], 1)
            self.assertEqual(tensor['shape'],  [-1,metadata['nchan'],metadata['nstation'],metadata['npol'],metadata['nstation'],metadata['npol']])
            self.assertEqual(tensor['dtype'],  'cf32')
            self.assertEqual(tensor['labels'], ['time', 'freq', 'station_i', 'pol_i', 'station_j', 'pol_j'])
            self.assertEqual(tensor['scales'], [[0, nreduce_time / metadata['chan_bw']], [metadata['cfreq'], metadata['chan_bw']], None, None, None, None])
            pol_units = ['X', 'Y']
            self.assertEqual(tensor['units'],  ['s', 'Hz', None, pol_units, None, pol_units])
        def check_data(ispan):
            pass
        with bf.Pipeline() as pipeline:
            data = CorrelateTestInputBlock([metadata], gulp_nframe=gulp_nframe)
            data = copy(data, space='cuda')
            data = bf.blocks.correlate(data, nreduce_time)
            ref1 = {}
            CallbackBlock(data, check_sequence, check_data, data_ref=ref1)
            data = bf.blocks.convert_visibilities(data, 'matrix')
            ref2 = {}
            CallbackBlock(data, check_sequence, check_data, data_ref=ref2)
            pipeline.run()

        # Now we check the results
        # Note: This must match the values in CorrelateTestInputBlock (TODO: Refactor/clean this)
        i = np.arange(metadata['nstation'] * metadata['npol'] * 2) % 255 - 127
        input_data = np.empty((metadata['nchan'], metadata['nstation'] * metadata['npol']), dtype=np.complex64)
        input_data[...].real = i[0::2]
        input_data[...].imag = i[1::2]
        expected_data = nreduce_time * input_data[:,:,None].conj() * input_data[:,None,:]
        expected_data_shaped = expected_data.reshape(
            (1, metadata['nchan'],
             metadata['nstation'], metadata['npol'],
             metadata['nstation'], metadata['npol']))

        # Check full matrix output
        idata = ref2['idata']
        idata = idata.copy('system')
        np.testing.assert_allclose(idata, expected_data_shaped, RTOL, ATOL)

        # Check lower-tri matrix output
        triu = np.triu_indices(metadata['nstation'] * metadata['npol'], 1)
        expected_data[..., triu[0], triu[1]] = 0
        expected_data = expected_data.reshape(
            (1, metadata['nchan'],
             metadata['nstation'], metadata['npol'],
             metadata['nstation'], metadata['npol']))
        idata = ref1['idata']
        idata = idata.copy('system')
        idata = idata.reshape(1, metadata['nchan'],
                              metadata['nstation']*metadata['npol'],
                              metadata['nstation']*metadata['npol'])
        # TODO: Assignment to the upper triangle of a Bifrost ndarray is
        #         silently failing! This is a WAR to use plain numpy instead.
        idata = np.array(idata)
        # Note: This is necessary because the upper triangle is left
        #         untouched by the kernel and ends up containing old
        #         data from the ring.
        idata[..., triu[0], triu[1]] = 0
        idata = idata.reshape(
            (1, metadata['nchan'],
             metadata['nstation'], metadata['npol'],
             metadata['nstation'], metadata['npol']))
        np.testing.assert_allclose(idata, expected_data, RTOL, ATOL)

    def test_convert_visibilities(self):
        gulp_nframe  = 100
        nreduce_time = 1000
        metadata = {
            'ntime':  10000,
            'nchan':    128,
            'nstation':  60,
            'npol':       2,
            'chan_bw': 25e3,
            'cfreq':   50e6
        }
        def check_sequence(seq):
            tensor = seq.header['_tensor']
            self.assertEqual(seq.header['gulp_nframe'], 1)
            self.assertEqual(tensor['shape'],  [-1,metadata['nchan'],metadata['nstation'],metadata['npol'],metadata['nstation'],metadata['npol']])
            self.assertEqual(tensor['dtype'],  'cf32')
            self.assertEqual(tensor['labels'], ['time', 'freq', 'station_i', 'pol_i', 'station_j', 'pol_j'])
            self.assertEqual(tensor['scales'], [[0, nreduce_time / metadata['chan_bw']], [metadata['cfreq'], metadata['chan_bw']], None, None, None, None])
            pol_units = ['X', 'Y']
            self.assertEqual(tensor['units'],  ['s', 'Hz', None, pol_units, None, pol_units])
        def check_data(ispan):
            pass
        with bf.Pipeline() as pipeline:
            data = CorrelateTestInputBlock([metadata], gulp_nframe=gulp_nframe)
            data = copy(data, space='cuda')
            data = bf.blocks.correlate(data, nreduce_time)
            data1 = bf.blocks.convert_visibilities(data, 'matrix')
            ref1 = {}
            CallbackBlock(data1, check_sequence, check_data, data_ref=ref1)
            data2 = bf.blocks.convert_visibilities(data, 'storage')
            data2 = bf.blocks.convert_visibilities(data2, 'matrix')
            ref2 = {}
            CallbackBlock(data2, check_sequence, check_data, data_ref=ref2)
            pipeline.run()
        expected_data = ref1['idata'].copy('system')
        actual_data   = ref2['idata'].copy('system')
        np.testing.assert_allclose(actual_data, expected_data, RTOL, ATOL)
