/*
 * Copyright (c) 2019-2023, The Bifrost Authors. All rights reserved.
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
#include <bifrost/io.h>
#include <bifrost/packet_writer.h>
#include "proclog.hpp"
#include "Socket.hpp"
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

#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <chrono>
#include <time.h>

#ifndef BF_SEND_NPKTBURST
#define BF_SEND_NPKTBURST 16
#endif

class RateLimiter {
  uint32_t _rate;
  uint64_t _counter;
  bool     _first;
  std::chrono::high_resolution_clock::time_point _start;
  std::chrono::high_resolution_clock::time_point _stop;
public:
  RateLimiter(uint32_t rate_limit=0)
   : _rate(rate_limit), _counter(0), _first(true) {}
  inline void set_rate(uint32_t rate_limit) { _rate = rate_limit; }
  inline uint32_t get_rate() { return _rate; }
  inline void reset_counter() { _first = true; _counter = 0; }
  inline void begin() {
    if( _first ) {
      _start = std::chrono::high_resolution_clock::now();
      _first = false;
    }
  }
  inline void end_and_wait(size_t npackets) {
    if( _rate > 0 ) {
      _stop = std::chrono::high_resolution_clock::now();
      _counter += std::max((size_t) 0, npackets);
      double elapsed_needed = (double) _counter / _rate;
      std::chrono::duration<double> elapsed_actual = std::chrono::duration_cast<std::chrono::duration<double>>(_stop-_start);
      
      double sleep_needed = elapsed_needed - elapsed_actual.count();
      if( sleep_needed > 0.001 ) {
        timespec sleep;
        sleep.tv_sec = (int) sleep_needed;
        sleep.tv_nsec = (int) ((sleep_needed - sleep.tv_sec)*1e9);
        nanosleep(&sleep, NULL);
      }
    }
  }
};

class PacketWriterMethod: public BoundThread {
protected:
    int         _fd;
    size_t      _max_burst_size;
    RateLimiter _limiter;
    int         _core;
public:
    PacketWriterMethod(int fd, size_t max_burst_size=BF_SEND_NPKTBURST, int core=-1)
     : BoundThread(core), _fd(fd), _max_burst_size(max_burst_size), _limiter(0), _core(core) {}
    virtual ssize_t send_packets(char* hdrs, 
                                 int   hdr_size,
                                 char* data, 
                                 int   data_size, 
                                 int   npackets,
                                 int   flags=0) {
        return 0;
    }
    virtual const char* get_name()     { return "generic_writer"; }
    inline void set_rate(uint32_t rate_limit) { _limiter.set_rate(rate_limit); }
    inline uint32_t get_rate() { return _limiter.get_rate(); }
    inline void reset_counter() { _limiter.reset_counter(); }
};

class DiskPacketWriter : public PacketWriterMethod {
public:
    DiskPacketWriter(int fd, size_t max_burst_size=BF_SEND_NPKTBURST, int core=-1)
     : PacketWriterMethod(fd, max_burst_size, core) {}
    ssize_t send_packets(char* hdrs, 
                         int   hdr_size,
                         char* data, 
                         int   data_size, 
                         int   npackets,
                         int   flags=0) {
        int i = 0;
        ssize_t status, nsend, nsent = 0;
        while(npackets > 0) {
            _limiter.begin();
            if( _max_burst_size > 0 ) {
                nsend = std::min(_max_burst_size, (size_t) npackets);
            } else {
                nsend = npackets;
            }
            for(int j=0; j<nsend; j++) {
                status = ::write(_fd, hdrs+hdr_size*(i+j), hdr_size);
                if( status != hdr_size ) continue;
                status = ::write(_fd, data+data_size*(i+j), data_size);
                if( status != data_size ) continue;
                nsent += 1;
            }
            i += nsend;
            npackets -= nsend;
            _limiter.end_and_wait(nsend);
        }
        return nsent;
    }
    inline const char* get_name() { return "disk_writer"; }
};

class UDPPacketSender : public PacketWriterMethod {
  int      _last_count;
  mmsghdr* _mmsg;
  iovec*   _iovs;
public:
    UDPPacketSender(int fd, size_t max_burst_size=BF_SEND_NPKTBURST, int core=-1)
     : PacketWriterMethod(fd, max_burst_size, core), _last_count(0), _mmsg(NULL), _iovs(NULL) {}
    ~UDPPacketSender() {
      if( _mmsg ) {
        free(_mmsg);
      }
      if( _iovs ) {
        free(_iovs);
      }
    }
    ssize_t send_packets(char* hdrs, 
                         int   hdr_size,
                         char* data, 
                         int   data_size, 
                         int   npackets,
                         int   flags=0) {
        if( npackets > _last_count ) {
          if( _mmsg ) {
            ::munlock(_mmsg, sizeof(struct mmsghdr)*_last_count);
            free(_mmsg);
          }
          if( _iovs ) {
            ::munlock(_iovs, sizeof(struct iovec)*2*_last_count);
            free(_iovs);
          }
          
          _last_count = npackets;
          _mmsg = (struct mmsghdr *) malloc(sizeof(struct mmsghdr)*npackets);
          _iovs = (struct iovec *) malloc(sizeof(struct iovec)*2*npackets);
          ::mlock(_mmsg, sizeof(struct mmsghdr)*npackets);
          ::mlock(_iovs, sizeof(struct iovec)*2*npackets);
          
          ::memset(_mmsg, 0, sizeof(struct mmsghdr)*npackets);
          
          for(int i=0; i<npackets; i++) {
            _mmsg[i].msg_hdr.msg_iov = &_iovs[2*i];
            _mmsg[i].msg_hdr.msg_iovlen = 2;
          }
        }
        
        for(int i=0; i<npackets; i++) {
            _iovs[2*i+0].iov_base = (hdrs + i*hdr_size);
            _iovs[2*i+0].iov_len = hdr_size;
            _iovs[2*i+1].iov_base = (data + i*data_size);
            _iovs[2*i+1].iov_len = data_size;
        }
        
        int i = 0;
        ssize_t nsend, nsent_batch, nsent = 0;
        while(npackets > 0) {
            _limiter.begin();
            if( _max_burst_size > 0 ) {
                nsend = std::min(_max_burst_size, (size_t) npackets);
            } else {
                nsend = npackets;
            }
            nsent_batch = ::sendmmsg(_fd, _mmsg+i, nsend, flags);
            if( nsent_batch > 0 ) {
                nsent += nsent_batch;
            }
            /*
            if( nsent_batch == -1 ) {
                std::cout << "sendmmsg failed: " << std::strerror(errno) << " with " << hdr_size << " and " << data_size << std::endl;
            }
            */
            i += nsend;
            npackets -= nsend;
            _limiter.end_and_wait(nsend);
        }
        
        return nsent;
    }
    inline const char* get_name() { return "udp_transmit"; }
};

#if defined BF_VERBS_ENABLED && BF_VERBS_ENABLED
#include "ib_verbs_send.hpp"

class UDPVerbsSender : public PacketWriterMethod {
    VerbsSend       _ibv;
    bf_comb_udp_hdr _udp_hdr;
    int             _last_size;
    int             _last_count;
    mmsghdr*        _mmsg;
    iovec*          _iovs;
public:
    UDPVerbsSender(int fd, size_t max_burst_size=BF_VERBS_SEND_NPKTBURST, int core=-1)
        : PacketWriterMethod(fd, max_burst_size, core), _ibv(fd, JUMBO_FRAME_SIZE), _last_size(0),
          _last_count(0), _mmsg(NULL), _iovs(NULL) {}
    ~UDPVerbsSender() {
      if( _mmsg ) {
        free(_mmsg);
      }
      if( _iovs ) {
        free(_iovs);
      }
    }
    ssize_t send_packets(char* hdrs, 
                         int   hdr_size,
                         char* data, 
                         int   data_size, 
                         int   npackets,
                         int   flags=0) {
        if( npackets > _last_count ) {
          if( _mmsg ) {
            ::munlock(_mmsg, sizeof(struct mmsghdr)*_last_count);
            free(_mmsg);
          }
          if( _iovs ) {
            ::munlock(_iovs, sizeof(struct iovec)*3*_last_count);
            free(_iovs);
          }
          
          _last_count = npackets;
          _mmsg = (struct mmsghdr *) malloc(sizeof(struct mmsghdr)*npackets);
          _iovs = (struct iovec *) malloc(sizeof(struct iovec)*3*npackets);
          ::mlock(_mmsg, sizeof(struct mmsghdr)*npackets);
          ::mlock(_iovs, sizeof(struct iovec)*3*npackets);
          
          ::memset(_mmsg, 0, sizeof(struct mmsghdr)*npackets);
          
          for(int i=0; i<npackets; i++) {
              _mmsg[i].msg_hdr.msg_iov = &_iovs[3*i];
              _mmsg[i].msg_hdr.msg_iovlen = 3;
          }
        }
        
        if( (hdr_size + data_size) != _last_size ) {
            _last_size = hdr_size + data_size;
            _ibv.get_ethernet_header(&(_udp_hdr.ethernet));
            _ibv.get_ipv4_header(&(_udp_hdr.ipv4), _last_size);
            _ibv.get_udp_header(&(_udp_hdr.udp), _last_size);
            
            if( _limiter.get_rate() > 0 ) {
                _ibv.set_rate_limit(_limiter.get_rate()*_last_size, _last_size, _max_burst_size);
            }
        }
        
        for(int i=0; i<npackets; i++) {
            _iovs[3*i+0].iov_base = &_udp_hdr;
            _iovs[3*i+0].iov_len = sizeof(bf_comb_udp_hdr);
            _iovs[3*i+1].iov_base = (hdrs + i*hdr_size);
            _iovs[3*i+1].iov_len = hdr_size;
            _iovs[3*i+2].iov_base = (data + i*data_size);
            _iovs[3*i+2].iov_len = data_size;
        }
        
        ssize_t nsent = _ibv.sendmmsg(_mmsg, npackets, flags);
        /*
        if( nsent == -1 ) {
            std::cout << "sendmmsg failed: " << std::strerror(errno) << " with " << hdr_size << " and " << data_size << std::endl;
        }
        */
        
        return nsent;
    }
    inline const char* get_name() { return "udp_verbs_transmit"; }
};
#endif // BF_VERBS_ENABLED

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
    PacketWriterThread(PacketWriterMethod* method, int core=-1)
     : BoundThread(core), _method(method), _core(core) {
        this->reset_stats();
    }
    inline void set_rate_limit(uint32_t rate_limit) { _method->set_rate(rate_limit); }
    inline uint32_t get_rate_limit() { return _method->get_rate(); }
    inline void reset_rate_limit_counter() { _method->reset_counter(); }
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
            _stats.nvalid += nsent;
            _stats.nvalid_bytes += nsent * (hdr_size + data_size);
            _stats.ninvalid += (npackets - nsent);
            _stats.ninvalid_bytes += (npackets - nsent) * (hdr_size + data_size);
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
        ::memset(&_desc, 0, sizeof(PacketDesc));
    }
    inline PacketDesc* get_description()            { return &_desc;                 }
    inline void set_nsrc(int nsrc)                  { _desc.nsrc = nsrc;             }
    inline void set_nchan(int nchan)                { _desc.nchan = nchan;           }
    inline void set_chan0(int chan0)                { _desc.chan0 = chan0;           }
    inline void set_tuning(int tuning)              { _desc.tuning = tuning;         }
    inline void set_gain(uint16_t gain)             { _desc.gain = gain;             }
    inline void set_decimation(uint32_t decimation) { _desc.decimation = decimation; }
};  

class BFpacketwriter_impl {
protected:
    std::string         _name;
    PacketWriterThread* _writer;
    PacketHeaderFiller* _filler;
    int                 _nsamples;
    BFdtype             _dtype;
    
    ProcLog             _bind_log;
    ProcLog             _stat_log;
    pid_t               _pid;
    
    char*               _pkt_hdrs;
    int                 _last_size;
    int                 _last_count;
    BFoffset            _framecount;
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
                               int                 nsamples,
                               BFdtype             dtype)
        : _name(writer->get_name()), _writer(writer), _filler(filler),
          _nsamples(nsamples), _dtype(dtype),
          _bind_log(_name+"/bind"),
          _stat_log(_name+"/stats"),
          _pkt_hdrs(NULL), _last_size(0), _last_count(0),
          _framecount(0) {
        _bind_log.update() << "ncore : " << 1 << "\n"
                           << "core0 : " << _writer->get_core() << "\n";
    }
    inline ~BFpacketwriter_impl() {
      if(_pkt_hdrs) {
        free(_pkt_hdrs);
      }
    }
    inline void set_rate_limit(uint32_t rate_limit) { _writer->set_rate_limit(rate_limit); }
    inline void reset_rate_limit_counter() { _writer->reset_rate_limit_counter(); }
    inline void reset_counter() { _framecount = 0; }
    BFstatus send(BFheaderinfo   info,
                  BFoffset       seq,
                  BFoffset       seq_increment,
                  BFoffset       src,
                  BFoffset       src_increment,
                  BFarray const* in);
};

class BFpacketwriter_generic_impl : public BFpacketwriter_impl {
    ProcLog            _type_log;
public:
    inline BFpacketwriter_generic_impl(PacketWriterThread* writer,
                                       int                 nsamples)
        : BFpacketwriter_impl(writer, nullptr, nsamples, BF_DTYPE_U8),
          _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new PacketHeaderFiller();
        _type_log.update("type : %s\n", "generic");
    }
};

class BFpacketwriter_simple_impl : public BFpacketwriter_impl {
    ProcLog            _type_log;
public:
    inline BFpacketwriter_simple_impl(PacketWriterThread* writer,
                                     int                 nsamples)
        : BFpacketwriter_impl(writer, nullptr, nsamples, BF_DTYPE_CI16),
          _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new SIMPLEHeaderFiller();
        _type_log.update("type : %s\n", "simple");
    }
};

class BFpacketwriter_chips_impl : public BFpacketwriter_impl {
    ProcLog            _type_log;
public:
    inline BFpacketwriter_chips_impl(PacketWriterThread* writer,
                                     int                 nsamples)
        : BFpacketwriter_impl(writer, nullptr, nsamples, BF_DTYPE_CI4),
          _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new CHIPSHeaderFiller();
        _type_log.update("type : %s\n", "chips");
    }
};

template<uint8_t B>
class BFpacketwriter_ibeam_impl : public BFpacketwriter_impl {
    uint8_t            _nbeam = B;
    ProcLog            _type_log;
public:
    inline BFpacketwriter_ibeam_impl(PacketWriterThread* writer,
                                     int                 nsamples)
        : BFpacketwriter_impl(writer, nullptr, nsamples, BF_DTYPE_CF32),
          _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new IBeamHeaderFiller<B>();
        _type_log.update("type : %s%i\n", "ibeam", _nbeam);
    }
};

template<uint8_t B>
class BFpacketwriter_pbeam_impl : public BFpacketwriter_impl {
    uint8_t            _nbeam = B;
    ProcLog            _type_log;
public:
    inline BFpacketwriter_pbeam_impl(PacketWriterThread* writer,
                                     int                 nsamples)
        : BFpacketwriter_impl(writer, nullptr, nsamples, BF_DTYPE_F32),
          _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new PBeamHeaderFiller<B>();
        _type_log.update("type : %s%i\n", "pbeam", _nbeam);
    }
};

class BFpacketwriter_cor_impl : public BFpacketwriter_impl {
    ProcLog            _type_log;
public:
    inline BFpacketwriter_cor_impl(PacketWriterThread* writer,
                                   int                 nsamples)
        : BFpacketwriter_impl(writer, nullptr, nsamples, BF_DTYPE_CF32),
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
     : BFpacketwriter_impl(writer, nullptr, nsamples, BF_DTYPE_CI8),
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
     : BFpacketwriter_impl(writer, nullptr, nsamples, BF_DTYPE_CI4),
       _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new DRXHeaderFiller();
        _type_log.update("type : %s\n", "drx");
    }
};

class BFpacketwriter_drx8_impl : public BFpacketwriter_impl {
    ProcLog            _type_log;
public:
    inline BFpacketwriter_drx8_impl(PacketWriterThread* writer,
                                    int                 nsamples)
     : BFpacketwriter_impl(writer, nullptr, nsamples, BF_DTYPE_CI8),
       _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new DRX8HeaderFiller();
        _type_log.update("type : %s\n", "drx8");
    }
};

class BFpacketwriter_tbf_impl : public BFpacketwriter_impl {
    int16_t            _nstand;
    ProcLog            _type_log;
public:
    inline BFpacketwriter_tbf_impl(PacketWriterThread* writer,
                                   int                 nsamples)
     : BFpacketwriter_impl(writer, nullptr, nsamples, BF_DTYPE_CI4),
       _nstand(0), _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new TBFHeaderFiller();
        _nstand = nsamples / 2 / 12;
        _type_log.update("type : %s%i\n", "tbf", _nstand);
    }
};

class BFpacketwriter_vbeam_impl : public BFpacketwriter_impl {
    ProcLog            _type_log;
public:
    inline BFpacketwriter_vbeam_impl(PacketWriterThread* writer,
                                   int                 nsamples)
     : BFpacketwriter_impl(writer, nullptr, nsamples, BF_DTYPE_CF32),
       _type_log((std::string(writer->get_name())+"/type").c_str()) {
        _filler = new VBeamHeaderFiller();
        _type_log.update("type : %s\n", "tbf");
    }
};

BFstatus BFpacketwriter_create(BFpacketwriter* obj,
                               const char*     format,
                               int             fd,
                               int             core,
                               BFiomethod      backend) {
    BF_ASSERT(obj, BF_STATUS_INVALID_POINTER);
    
    int nsamples = 0;
    if(std::string(format).substr(0, 8) == std::string("generic_") ) {
        nsamples = std::atoi((std::string(format).substr(8, std::string(format).length())).c_str());
    } else if( std::string(format).substr(0, 6) == std::string("simple") ) {
        nsamples = 2048;
    } else if( std::string(format).substr(0, 6) == std::string("chips_") ) {
        int nchan = std::atoi((std::string(format).substr(6, std::string(format).length())).c_str());
        nsamples = 32*nchan;
    } else if( std::string(format).substr(0, 5) == std::string("ibeam") ) {
        int nbeam = std::stoi(std::string(format).substr(5, 1));
        int nchan = std::atoi((std::string(format).substr(7, std::string(format).length())).c_str());
        nsamples = 2*nbeam*nchan;
    } else if( std::string(format).substr(0, 5) == std::string("pbeam") ) {
        int nchan = std::atoi((std::string(format).substr(7, std::string(format).length())).c_str());
        nsamples = 4*nchan;
    } else if( std::string(format).substr(0, 4) == std::string("cor_") ) {
        int nchan = std::atoi((std::string(format).substr(4, std::string(format).length())).c_str());
        nsamples = 4*nchan;
    } else if( format == std::string("tbn") ) {
        nsamples = 512;
    } else if( format == std::string("drx") ) {
        nsamples = 4096;
    } else if( format == std::string("drx8") ) {
        nsamples = 4096;
    } else if( format == std::string("tbf") ) {
        nsamples = 6144;
    } else if( std::string(format).substr(0, 6) == std::string("vbeam_") ) {
        // e.g. "vbeam_184" is a 184-channel voltage beam"
        int nchan = std::atoi((std::string(format).substr(13, std::string(format).length())).c_str());
        nsamples = 2*nchan; // 2 polarizations. Natively 32-bit floating complex (see implementation class)
    }
    
    PacketWriterMethod* method;
    if( backend == BF_IO_DISK ) {
        method = new DiskPacketWriter(fd, core=core);
    } else if( backend == BF_IO_UDP ) {
        method = new UDPPacketSender(fd, core=core);
#if defined BF_VERBS_ENABLED && BF_VERBS_ENABLED
    } else if( backend == BF_IO_VERBS ) {
        method = new UDPVerbsSender(fd, core=core);
#endif
    } else {
        return BF_STATUS_UNSUPPORTED;
    }
    PacketWriterThread* writer = new PacketWriterThread(method, core);
    
    if( std::string(format).substr(0, 8) == std::string("generic_") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_generic_impl(writer, nsamples),
                           *obj = 0);
    } else if( std::string(format).substr(0, 6) == std::string("simple") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_simple_impl(writer, nsamples),
                           *obj = 0);
    } else if( std::string(format).substr(0, 6) == std::string("chips_") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_chips_impl(writer, nsamples),
                           *obj = 0);
#define MATCH_IBEAM_MODE(NBEAM) \
    } else if( std::string(format).substr(0, 7) == std::string("ibeam"#NBEAM"_") ) { \
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_ibeam_impl<NBEAM>(writer, nsamples), \
                           *obj = 0);
    MATCH_IBEAM_MODE(1)
    MATCH_IBEAM_MODE(2)
    MATCH_IBEAM_MODE(3)
    MATCH_IBEAM_MODE(4)
#undef MATCH_IBEAM_MODE
#define MATCH_PBEAM_MODE(NBEAM) \
    } else if( std::string(format).substr(0, 7) == std::string("pbeam"#NBEAM"_") ) { \
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_pbeam_impl<NBEAM>(writer, nsamples), \
                           *obj = 0);
    MATCH_PBEAM_MODE(1)
#undef MATCH_PBEAM_MODE
    } else if( std::string(format).substr(0, 4) == std::string("cor_") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_cor_impl(writer, nsamples),
                           *obj = 0);
    } else if( format == std::string("tbn") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_tbn_impl(writer, nsamples),
                           *obj = 0);
    } else if( format == std::string("drx") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_drx_impl(writer, nsamples),
                           *obj = 0);
    } else if( format == std::string("drx8") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_drx8_impl(writer, nsamples),
                           *obj = 0);
    } else if( std::string(format).substr(0, 3) == std::string("tbf")  ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_tbf_impl(writer, nsamples),
                           *obj = 0);
    } else if( std::string(format).substr(0, 6) == std::string("vbeam_") ) {
        BF_TRY_RETURN_ELSE(*obj = new BFpacketwriter_vbeam_impl(writer, nsamples),
                           *obj = 0);
    } else {
        return BF_STATUS_UNSUPPORTED;
    }
}
