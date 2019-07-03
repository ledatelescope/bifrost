/*
 * Copyright (c) 2019, The Bifrost Authors. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of The Bifrost Authors nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "assert.hpp"
#include <bifrost/packet_writer.h>
#include "proclog.hpp"
#include "formats/formats.hpp"
#include "utils.hpp"
#include "hw_locality.hpp"

#include <arpa/inet.h>  // For ntohs
#include <sys/socket.h> // For recvfrom

#include <queue>
#include <memory>
#include <stdexcept>
#include <cstdlib>      // For posix_memalign
#include <cstring>      // For memcpy, memset
#include <cstdint>

#include <sys/types.h>
#include <unistd.h>
#include <fstream>

class PacketWriterMethod {
protected:
    int       _fd;
public:
    PacketWriterMethod(int fd)
     : _fd(fd) {}
    virtual ssize_t send_packets(char* hdrs, 
                                 int   hdr_size,
                                 char* data, 
                                 int   data_size, 
                                 int   npackets,
                                 int   flags=0) {
        return 0;
    }
    virtual const char* get_name()     { return "generic_writer"; }
};

struct PacketStats {
    size_t ninvalid;
    size_t ninvalid_bytes;
    size_t nlate;
    size_t nlate_bytes;
    size_t nvalid;
    size_t nvalid_bytes;
};

class PacketWriterThread : public BoundThread {
private:
    PacketWriterMethod*  _method;
    PacketStats          _stats;
    int                  _core;
    
public:
    PacketWriterThread(PacketWriterMethod* method, int core=0)
     : BoundThread(core), _method(method), _core(core) {
        this->reset_stats();
    }
    inline ssize_t send(char* hdrs,
                        int   hdr_size,
                        char* datas,
                        int   data_size,
                        int   npackets) {
        ssize_t nsent = _method->send_packets(hdrs, hdr_size, datas, data_size, npackets);
        if( nsent == -1 ) {
            _stats.ninvalid += npackets;
            _stats.ninvalid_bytes += npackets * (hdr_size + data_size);
        } else {
            _stats.nvalid += npackets;
            _stats.nvalid_bytes += npackets * (hdr_size + data_size);
        }
        return nsent;
    }
    inline const char* get_name() { return _method->get_name(); }
    inline const int get_core() { return _core; }
    inline const PacketStats* get_stats() const { return &_stats; }
    inline void reset_stats() {
        ::memset(&_stats, 0, sizeof(_stats));
    }
};

class BFheaderinfo_impl {
    PacketDesc         _desc;
public:
    inline BFheaderinfo_impl() {
        memset(&_desc, 0, sizeof(PacketDesc));
    }
    inline PacketDesc* get_description()            { return &_desc;                 }
    inline void set_nsrc(int nsrc)                  { _desc.nsrc = nsrc;             }
    inline void set_nchan(int nchan)                { _desc.nchan = nchan;           }
    inline void set_chan0(int chan0)                { _desc.chan0 = chan0;           }
    inline void set_tuning(int tuning)              { _desc.tuning = tuning;         }
    inline void set_gain(uint16_t gain)             { _desc.gain = gain;             }
    inline void set_decimation(uint16_t decimation) { _desc.decimation = decimation; }
};  

class BFpacketwriter_impl {
protected:
    std::string         _name;
    PacketWriterThread* _writer;
    PacketHeaderFiller* _filler;
    
    ProcLog             _bind_log;
    ProcLog             _stat_log;
    pid_t               _pid;
    
    int                 _nsamples;
private:
    void update_stats_log() {
        const PacketStats* stats = _writer->get_stats();
        _stat_log.update() << "ngood_bytes    : " << stats->nvalid_bytes << "\n"
                           << "nmissing_bytes : " << stats->ninvalid_bytes << "\n"
                           << "ninvalid       : " << stats->ninvalid << "\n"
                           << "ninvalid_bytes : " << stats->ninvalid_bytes << "\n"
                           << "nlate          : " << stats->nlate << "\n"
                           << "nlate_bytes    : " << stats->nlate_bytes << "\n"
                           << "nvalid         : " << stats->nvalid << "\n"
                           << "nvalid_bytes   : " << stats->nvalid_bytes << "\n";
    }
public:
    inline BFpacketwriter_impl(PacketWriterThread* writer, 
                               PacketHeaderFiller* filler,
                               int                 nsamples)
        : _name(writer->get_name()), _writer(writer), _filler(filler),
          _bind_log(_name+"/bind"),
          _stat_log(_name+"/stats"),
          _nsamples(nsamples) {
        _bind_log.update() << "ncore : " << 1 << "\n"
                           << "core0 : " << _writer->get_core() << "\n";
    }
    virtual ~BFpacketwriter_impl() {}
    BFstatus send(BFheaderinfo   desc,
                  BFoffset       seq,
                  BFoffset       seq_increment,
                  BFoffset       seq_stride,
                  BFoffset       src,
                  BFoffset       src_increment,
                  BFoffset       src_stride,
                  BFarray const* in);
};

class BFpacketwriter_generic_impl : public BFpacketwriter_impl {
    ProcLog            _type_log;
public:
    inline BFpacketwriter_generic_impl(PacketWriterThread* writer,
                                       int                 nsamples)
        : BFpacketwriter_impl(writer, nullptr, nsamples),
          _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new PacketHeaderFiller();
        _type_log.update("type : %s\n", "generic");
    }
};

class BFpacketwriter_chips_impl : public BFpacketwriter_impl {
    ProcLog            _type_log;
public:
    inline BFpacketwriter_chips_impl(PacketWriterThread* writer,
                                     int                 nsamples)
        : BFpacketwriter_impl(writer, nullptr, nsamples),
          _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new CHIPSHeaderFiller();
        _type_log.update("type : %s\n", "chips");
    }
};

class BFpacketwriter_cor_impl : public BFpacketwriter_impl {
    ProcLog            _type_log;
public:
    inline BFpacketwriter_cor_impl(PacketWriterThread* writer,
                                   int                 nsamples)
        : BFpacketwriter_impl(writer, nullptr, nsamples),
          _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new CORHeaderFiller();
        _type_log.update("type : %s\n", "cor");
    }
};

class BFpacketwriter_tbn_impl : public BFpacketwriter_impl {
    ProcLog            _type_log;
public:
    inline BFpacketwriter_tbn_impl(PacketWriterThread* writer,
                                   int                 nsamples)
     : BFpacketwriter_impl(writer, nullptr, nsamples),
       _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new TBNHeaderFiller();
        _type_log.update("type : %s\n", "tbn");
    }
};

class BFpacketwriter_drx_impl : public BFpacketwriter_impl {
    ProcLog            _type_log;
public:
    inline BFpacketwriter_drx_impl(PacketWriterThread* writer,
                                   int                 nsamples)
     : BFpacketwriter_impl(writer, nullptr, nsamples),
       _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new DRXHeaderFiller();
        _type_log.update("type : %s\n", "drx");
    }
};

class BFpacketwriter_tbf_impl : public BFpacketwriter_impl {
    ProcLog            _type_log;
public:
    inline BFpacketwriter_tbf_impl(PacketWriterThread* writer,
                                   int                 nsamples)
     : BFpacketwriter_impl(writer, nullptr, nsamples),
       _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new TBFHeaderFiller();
        _type_log.update("type : %s\n", "tbf");
    }
};
