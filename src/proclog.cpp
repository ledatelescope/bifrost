/*
 * Copyright (c) 2016-2021, The Bifrost Authors. All rights reserved.
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

#include <bifrost/config.h>
#include <bifrost/proclog.h>
#include "trace.hpp"
#include "proclog.hpp"
#include "fileutils.hpp"

#include <fstream>
#include <cstdlib>     // For system
#include <cstdarg>     // For va_start, va_list, va_end
#include <sys/types.h> // For getpid
#include <dirent.h>    // For opendir, readdir, closedir
#include <unistd.h>    // For getpid
#include <system_error>
#include <set>
#include <mutex>

class ProcLogMgr {
	static constexpr const char* base_logdir = BF_PROCLOG_DIR;
	std::string            _logdir;
	std::set<std::string>  _logs;
	std::set<std::string>  _created_dirs;
	mutable std::mutex     _mutex;
	void try_base_logdir_cleanup() {
		// Do this with a file lock to avoid interference from other processes
		LockFile lock(std::string(base_logdir) + ".lock");
		DIR* dp;
		// Remove pid dirs for which a corresponding process does not exist
		if( (dp = opendir(base_logdir)) ) {
			struct dirent* ep;
			while( (ep = readdir(dp)) ) {
				pid_t pid = atoi(ep->d_name);
				if( pid && !process_exists(pid) ) {
					try {
						remove_files_recursively(std::string(base_logdir) + "/" +
									 std::to_string(pid));
					} catch( std::exception& ) {}
				}
			}
			closedir(dp);
		}
		// Remove the base_logdir if it's empty
		try { remove_dir(base_logdir); }
		catch( std::exception& ) {}
	}
	ProcLogMgr()
		: _logdir(std::string(base_logdir) + "/" + std::to_string(getpid())) {
		this->try_base_logdir_cleanup();
		make_dir(base_logdir, 0777);
		make_dir(_logdir);
	}
	~ProcLogMgr() {
		try {
			remove_files_recursively(_logdir);
			this->try_base_logdir_cleanup();
		} catch( std::exception& ) {}
	}
	FILE* open_file(std::string filename) {
		// Note: Individual log dirs and files are only created if they are
		//         actually updated with data.
		// Note: name may contain subdirs, so we must ensure they exist first
		this->ensure_dir_exists(filename);
		FILE* logfile = (FILE*)fopen(filename.c_str(), "w");
		if( !logfile ) {
			throw std::runtime_error("fopen(\""+filename+"\", \"w\") failed");
		}
		return logfile;
	}
public:
	ProcLogMgr(ProcLogMgr& ) = delete;
	ProcLogMgr& operator=(ProcLogMgr& ) = delete;
	static ProcLogMgr& get() {
		static ProcLogMgr proclog;
		return proclog;
	}
	void ensure_dir_exists(std::string filename) {
		std::string dirname = get_dirname(filename);
		if( !_created_dirs.count(dirname) ) {
			// Note: make_dir is really slow, so we ensure we call it only once
			make_dir(dirname);
			_created_dirs.insert(dirname);
		}
	}
	std::string create_log(std::string name) {
		std::lock_guard<std::mutex> lock(_mutex);
		std::string origname = name;
		std::string modname = name.substr(0, name.find("/"));
		std::string logname = name.substr(name.find("/"), name.length());
		std::string filename = _logdir + "/" + name;
		
		int i = 1;
		while( _logs.count(filename) ) {
			// Disambiguate by adding suffix to name
			name = modname + "_" + std::to_string(++i);
			if( logname.length() > 0 ) {
				name = name + "/" + logname;
			}
			filename = _logdir + "/" + name;
		}
		_logs.insert(filename);
		return filename;
	}
	void destroy_log(std::string filename) {
		std::lock_guard<std::mutex> lock(_mutex);
		remove_file(filename);
		_logs.erase(filename);
	}
	void update_log_s(std::string filename, const char* str) {
		std::lock_guard<std::mutex> lock(_mutex);
		FILE* logfile = this->open_file(filename);
		fputs(str, logfile);
		fclose(logfile);
	}
	void update_log_v(std::string filename, const char* fmt, va_list args) {
		std::lock_guard<std::mutex> lock(_mutex);
		FILE* logfile = this->open_file(filename);
		vfprintf(logfile, fmt, args);
		fclose(logfile);
	}
};

BFproclog_impl::BFproclog_impl(std::string name)
	: _filename(ProcLogMgr::get().create_log(name)) {}
BFproclog_impl::~BFproclog_impl() {
	ProcLogMgr::get().destroy_log(_filename);
}
void BFproclog_impl::update_s(const char* str) {
	ProcLogMgr::get().update_log_s(_filename, str);
}
void BFproclog_impl::update_v(const char* fmt, va_list args) {
	ProcLogMgr::get().update_log_v(_filename, fmt, args);
}
void BFproclog_impl::update(const char* fmt, ...) {
	va_list args;
	va_start(args, fmt);
	this->update_v(fmt, args);
	va_end(args);
}
movable_ofstream_WAR BFproclog_impl::update() {
	ProcLogMgr::get().ensure_dir_exists(_filename);
	// TODO: gcc < 5 has a bug where std streams are not movable
	//return std::ofstream(_filename);
	return movable_ofstream_WAR(_filename);
}

BFstatus bfProcLogCreate(BFproclog* log_ptr, const char* name) {
	BF_ASSERT(log_ptr, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(name,    BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN_ELSE(*log_ptr = new BFproclog_impl(name),
	                   *log_ptr = 0);
}
BFstatus bfProcLogDestroy(BFproclog log) {
	BF_ASSERT(log, BF_STATUS_INVALID_HANDLE);
	delete log;
	return BF_STATUS_SUCCESS;
}
BFstatus bfProcLogUpdate(BFproclog log, const char* str) {
	BF_ASSERT(log, BF_STATUS_INVALID_HANDLE);
	BF_ASSERT(str, BF_STATUS_INVALID_POINTER);
	BF_TRY_RETURN(log->update_s(str));
}
