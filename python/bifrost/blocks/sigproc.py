
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

from __future__ import absolute_import

from bifrost.pipeline import SourceBlock, SinkBlock
import bifrost.sigproc2 as sigproc
from bifrost.DataType import DataType
from bifrost.units import convert_units
from numpy import transpose

from copy import deepcopy
import os

def _get_with_default(obj, key, default=None):
    return obj[key] if key in obj else default

def _mjd2unix(mjd):
    return (mjd - 40587) * 86400

def _unix2mjd(unix):
    return unix / 86400. + 40587

class SigprocSourceBlock(SourceBlock):
    def __init__(self, filenames, gulp_nframe, unpack=True, *args, **kwargs):
        super(SigprocSourceBlock, self).__init__(filenames, gulp_nframe, *args, **kwargs)
        self.unpack = unpack
    def create_reader(self, sourcename):
        return sigproc.SigprocFile(sourcename)
    def on_sequence(self, ireader, sourcename):
        ihdr = ireader.header
        assert(ihdr['data_type'] in [1,  # filterbank
                                     2,  # (dedispersed) time series
                                     6]) # dedispersed subbands
        for coord_frame in ['pulsarcentric', 'barycentric', 'topocentric']:
            if coord_frame in ihdr and bool(ihdr[coord_frame]):
                break
        tstart_unix = _mjd2unix(ihdr['tstart'])
        nbit = ihdr['nbits']
        if self.unpack:
            nbit = max(nbit, 8)
        ohdr = {
            '_tensor': {
                'dtype':  ['u', 'i'][ihdr['signed']] + str(nbit),
                'shape':  [-1, ihdr['nifs'], ihdr['nchans']],
                'labels': ['time', 'pol', 'freq'],
                'scales': [(tstart_unix, ihdr['tsamp']),
                           None,
                           (ihdr['fch1'], ihdr['foff'])],
                'units':  ['s', None, 'MHz']
            },
            'frame_rate': 1. / ihdr['tsamp'], # TODO: Used for anything?
            'source_name':   _get_with_default(ihdr, 'source_name'),
            'rawdatafile':   _get_with_default(ihdr, 'rawdatafile'),
            'az_start':      _get_with_default(ihdr, 'az_start'),
            'za_start':      _get_with_default(ihdr, 'za_start'),
            'raj':           _get_with_default(ihdr, 'src_raj'),
            'dej':           _get_with_default(ihdr, 'src_dej'),
            'refdm':         _get_with_default(ihdr, 'refdm', 0.),
            'refdm_units':   'pc cm^-3',
            'telescope':     _get_with_default(ihdr, 'telescope_id'),
            'machine':       _get_with_default(ihdr, 'machine_id'),
            'ibeam':         _get_with_default(ihdr, 'ibeam'),
            'nbeams':        _get_with_default(ihdr, 'nbeams'),
            'coord_frame':   coord_frame,
        }
        # Convert ids to strings
        ohdr['telescope'] = sigproc.id2telescope(ohdr['telescope'])
        ohdr['machine']   = sigproc.id2machine(ohdr['machine'])
        # Note: This gives 32 bits to the fractional part of a second,
        #         corresponding to ~0.233ns resolution. The whole part
        #         gets at least 31 bits, which will overflow in 2038.
        time_tag  = int(round(tstart_unix * 2**32))
        ohdr['time_tag'] = time_tag
        ohdr['name']     = sourcename
        return [ohdr]
    def on_data(self, reader, ospans):
        ospan = ospans[0]
        #print "SigprocReadBlock::on_data", ospan.data.dtype
        if self.unpack:
            indata = reader.read(ospan.shape[0])
            nframe = indata.shape[0]
            #print indata.shape, indata.dtype, nframe
            #print indata
            ospan.data[:nframe] = indata
            # TODO: This will break when frame size < 1 byte
            #         Can't use frame_nbyte; must use something like frame_nbit
            #           Gets tricky though because what array shape+dtype to use?
            #             Would need to use an array class that supports dtypes
            #               of any nbit, and deals with consequent indexing issues.
            #           Admittedly, it is probably a rare case, because a detected
            #             time series would typically have SNR warranting >= 8
            #             bits.
            #           Multiple pols could be included, but only if chan and pol
            #             dims are merged together.
            #print "NBYTE", nbyte
            #assert(nbyte % reader.frame_nbyte == 0)
            #nframe = nbyte // reader.frame_nbyte
        else:
            nbyte = reader.readinto(ospan.data)
            if nbyte % ospan.frame_nbyte:
                raise IOError("Input file is truncated")
            nframe = nbyte // ospan.frame_nbyte
        return [nframe]

def read_sigproc(filenames, gulp_nframe, unpack=True, *args, **kwargs):
    """Read SIGPROC data files.

    Capable of reading filterbank, time series, and dedispersed subband data.

    Args:
        filenames (list): List of input filenames.
        gulp_nframe (int): No. frames to read at a time.
        unpack (bool): If True, 1-4 bit data are unpacked to 8 bits.
        *args: Arguments to ``bifrost.pipeline.SourceBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.SourceBlock``.

    **Tensor semantics**::

        Output: ['time', 'pol', 'freq'], dtype = u/i*, space = SYSTEM

    Returns:
        SigprocSourceBlock: A new block instance.
    """
    return SigprocSourceBlock(filenames, gulp_nframe, unpack,
                              *args, **kwargs)

def _copy_item_if_exists(dst, src, key, newkey=None):
    if key in src:
        if newkey is None:
            newkey = key
        dst[newkey] = src[key]

class SigprocSinkBlock(SinkBlock):
    def __init__(self, iring, path=None, *args, **kwargs):
        super(SigprocSinkBlock, self).__init__(iring, *args, **kwargs)
        if path is None:
            path = ''
        self.path = path
    def on_sequence(self, iseq):
        ihdr = iseq.header
        itensor = ihdr['_tensor']

        axnames = list(itensor['labels'])
        shape   = list(itensor['shape'])
        scales  = list(itensor['scales'])
        units   = list(itensor['units'])
        ndim    = len(shape)
        dtype   = DataType(itensor['dtype'])

        sigproc_hdr = {}
        _copy_item_if_exists(sigproc_hdr, ihdr, 'source_name')
        _copy_item_if_exists(sigproc_hdr, ihdr, 'rawdatafile')
        _copy_item_if_exists(sigproc_hdr, ihdr, 'az_start')
        _copy_item_if_exists(sigproc_hdr, ihdr, 'za_start')
        _copy_item_if_exists(sigproc_hdr, ihdr, 'raj', 'src_raj')
        _copy_item_if_exists(sigproc_hdr, ihdr, 'dej', 'src_dej')
        if 'telescope' in ihdr:
            sigproc_hdr['telescope_id'] = sigproc.telescope2id(ihdr['telescope'])
        if 'machine' in ihdr:
            sigproc_hdr['machine_id'] = sigproc.machine2id(ihdr['machine'])
        _copy_item_if_exists(sigproc_hdr, ihdr, 'telescope_id')
        _copy_item_if_exists(sigproc_hdr, ihdr, 'machine_id')
        _copy_item_if_exists(sigproc_hdr, ihdr, 'ibeam')
        _copy_item_if_exists(sigproc_hdr, ihdr, 'nbeams')
        sigproc_hdr['nbits'] = dtype.itemsize_bits
        _copy_item_if_exists(sigproc_hdr, ihdr, 'barycentric')
        _copy_item_if_exists(sigproc_hdr, ihdr, 'pulsarcentric')
        if dtype.is_integer and dtype.is_signed:
            sigproc_hdr['signed'] = True
        if 'coord_frame' in ihdr:
            coord_frame = ihdr['coord_frame']
        else:
            coord_frame = None
        sigproc_hdr['pulsarcentric'] = (coord_frame == 'pulsarcentric')
        sigproc_hdr['barycentric']   = (coord_frame == 'barycentric')

        filename = os.path.join(self.path, ihdr['name'])

        if ndim >= 3 and axnames[-3:] == ('time', 'pol', 'freq'):
            self.data_format = 'filterbank'
            assert(dtype.is_real)
            sigproc_hdr['data_type'] = 1
            sigproc_hdr['nifs']   = shape[-2]
            sigproc_hdr['nchans'] = shape[-1]
            sigproc_hdr['tstart'] = _unix2mjd(scales[-3][0])
            sigproc_hdr['tsamp']  = convert_units(scales[-3][1], units[-3], 's')
            sigproc_hdr['fch1']   = convert_units(scales[-1][0], units[-1], 'MHz')
            sigproc_hdr['foff']   = convert_units(scales[-1][1], units[-1], 'MHz')
            if 'refdm' in ihdr:
                sigproc_hdr['refdm'] = convert_units(ihdr['refdm'],
                                                     ihdr['refdm_units'],
                                                     'pc cm^-3')
            if ndim == 3:
                filename += '.fil'
                self.ofile = open(filename, 'wb')
                sigproc.write_header(sigproc_hdr, self.ofile)
            elif ndim == 4:
                if axnames[-4] != 'beam':
                    raise ValueError("Expected first axis to be 'beam'"
                                     " got '%s'" % axnames[-4])
                nbeam = shape[-4]
                sigproc_hdr['nbeams'] = nbeam
                filenames = [filename + '.%06iof.%06i.fil' % (b + 1, nbeam)
                             for b in xrange(nbeam)]
                self.ofiles = [open(fname, 'wb') for fname in filenames]
                for b in xrange(nbeam):
                    sigproc_hdr['ibeam'] = b
                    sigproc.write_header(sigproc_hdr, self.ofiles[b])
            else:
                raise ValueError("Too many dimensions")

        elif ndim >= 2 and 'time' in axnames and 'pol' in axnames:
            pol_axis = axnames.index('pol')
            if pol_axis != ndim - 1:
                # Need to move pol axis
                # Note: We support this because it tends to be convenient
                #         for rest of the pipeline to operate with pol being
                #         the first dim, and doing the transpose on the fly
                #         inside this block is unlikely to cost much relative
                #         to disk perf (and it's free if npol==1).
                axnames.append(axnames[pol_axis]); del axnames[pol_axis]
                shape.append(shape[pol_axis]);     del shape[pol_axis]
                scales.append(scales[pol_axis]);   del scales[pol_axis]
                units.append(units[pol_axis]);     del units[pol_axis]
            self.pol_axis = pol_axis
            self.data_format = 'timeseries'
            assert(dtype.is_real)
            sigproc_hdr['data_type'] = 2
            sigproc_hdr['nchans'] = 1
            sigproc_hdr['nifs']   = shape[-2]
            sigproc_hdr['tstart'] = _unix2mjd(scales[-2][0])
            sigproc_hdr['tsamp']  = convert_units(scales[-2][1], units[-2], 's')
            if 'cfreq' in ihdr and 'bw' in ihdr:
                sigproc_hdr['fch1'] = convert_units(ihdr['cfreq'],
                                                    ihdr['cfreq_units'],
                                                    'MHz')
                sigproc_hdr['foff'] = convert_units(ihdr['bw'],
                                                    ihdr['bw_units'],
                                                    'MHz')
            # TODO: Write ndim separate output files, each with its own refdm
            if ndim == 2:
                if 'refdm' in ihdr:
                    sigproc_hdr['refdm'] = convert_units(ihdr['refdm'],
                                                         ihdr['refdm_units'],
                                                         'pc cm^-3')
                filename += '.tim'
                self.ofile = open(filename, 'wb')
                sigproc.write_header(sigproc_hdr, self.ofile)
            elif ndim == 3:
                if axnames[-3] != 'dispersion':
                    raise ValueError("Expected first axis to be 'dispersion'"
                                     " got '%s'" % axnames[-3])
                ndm = shape[-3]
                dm0 = scales[-3][0]
                ddm = scales[-3][1]
                dms = [dm0 + ddm * d for d in xrange(ndm)]
                dms = [convert_units(dm, units[-3], 'pc cm^-3') for dm in dms]
                filenames = [filename + '.%09.2f.tim' % dm for dm in dms]
                self.ofiles = [open(fname, 'wb') for fname in filenames]
                for d, dm in enumerate(dms):
                    sigproc_hdr['refdm'] = dm
                    sigproc.write_header(sigproc_hdr, self.ofiles[d])
            else:
                raise ValueError("Too many dimensions")

        elif ndim == 4 and axnames[-3:] == ('pol', 'freq', 'phase'):
            self.data_format = 'pulseprofile'
            assert(dtype.is_real)
            sigproc_hdr['data_type'] = 2
            sigproc_hdr['nifs']   = shape[-3]
            sigproc_hdr['nchans'] = shape[-2]
            sigproc_hdr['nbins']  = shape[-1]
            sigproc_hdr['tstart'] = _unix2mjd(scales[-4][0])
            sigproc_hdr['tsamp']  = convert_units(scales[-4][1], units[-4], 's')
            sigproc_hdr['fch1']   = convert_units(scales[-2][0], units[-2], 'MHz')
            sigproc_hdr['foff']   = convert_units(scales[-2][1], units[-2], 'MHz')
            if 'refdm' in ihdr:
                sigproc_hdr['refdm'] = convert_units(ihdr['refdm'],
                                                     ihdr['refdm_units'],
                                                     'pc cm^-3')
            _copy_item_if_exists(sigproc_hdr, ihdr, 'npuls')
            self.filename = filename
            self.sigproc_hdr = sigproc_hdr
            self.t0 = scales[-4][0]
            self.dt = scales[-4][1]

        else:
            raise ValueError("Axis labels do not correspond to a known data format: " +
                             str(axnames) + "\nKnown formats are:" +
                             "\n  [time, pol, freq]\n  [beam, time, pol]\n" +
                             "  [time, pol]\n  [dispersion, time, pol]\n" +
                             "  [pol, freq, phase]")

    def on_sequence_end(self, iseq):
        if hasattr(self, 'ofile'):
            self.ofile.close()
        elif hasattr(self, 'ofiles'):
            for ofile in self.ofiles:
                ofile.close()

    def on_data(self, ispan):
        idata = ispan.data
        if self.data_format == 'filterbank':
            if len(idata.shape) == 3:
                idata.tofile(self.ofile)
            else:
                for b in xrange(idata.shape[0]):
                    idata[b].tofile(self.ofiles[b])
        elif self.data_format == 'timeseries':
            ndim = len(idata.shape)
            if self.pol_axis != ndim - 1:
                perm = list(xrange(ndim))
                del perm[self.pol_axis]
                perm.append(self.pol_axis);
                idata = transpose(idata, perm)
            if ndim == 2:
                idata.tofile(self.ofile)
            else:
                for d in xrange(idata.shape[0]):
                    idata[d].tofile(self.ofiles[d])
        elif self.data_format == 'pulseprofile':
            time_unix = self.t0 + ispan.frame_offset * self.dt
            filename = self.filename + '.%017.6f.tim' % time_unix
            with open(filename, 'wb') as ofile:
                self.sigproc_hdr['tstart'] += self.sigproc_hdr['tsamp']
                sigproc.write_header(self.sigproc_hdr, ofile)
                idata.tofile(ofile)
        else:
            raise ValueError("Internal error: Unknown data format!")

def write_sigproc(iring, path=None, *args, **kwargs):
    """Write data as Sigproc files.

    Args:
        iring (Ring or Block): Input data source.
        path (str): Path specifying where to write output files.
        *args: Arguments to ``bifrost.pipeline.SinkBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.SinkBlock``.

    **Tensor semantics**::

        Input:  [time, pol, freq], dtype = any, space = SYSTEM
        Output: Filterbank, one file per sequence

        Input:  [beam, time, pol, freq], dtype = any, space = SYSTEM
        Output: Filterbank, one file per beam

        Input:  [time, pol], dtype = any, space = SYSTEM
        Output: Time series, one file per sequence

        Input:  [dispersion, time, pol], dtype = any, space = SYSTEM
        Output: Time series, one file per dispersion measure trial

        Input:  [pol, freq, phase], dtype = any, space = SYSTEM
        Output: Pulse profile, one file per frame

    Returns:
        SigprocSinkBlock: A new block instance.
    """
    return SigprocSinkBlock(iring, path, *args, **kwargs)
