/*
 * Copyright (c) 2016, The Bifrost Authors. All rights reserved.
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

#include <bifrost/proclog.h>
#include "trace.hpp"
#include "proclog.hpp"

#include <fstream>
#include <cstdlib>     // For system
#include <cstdarg>     // For va_start, va_list, va_end
#include <sys/file.h>  // For flock
#include <sys/stat.h>  // For fstat
#include <sys/types.h> // For getpid
#include <dirent.h>    // For opendir, readdir, closedir
#include <unistd.h>    // For getpid
#include <system_error>
#include <set>
#include <mutex>

void make_dir(std::string path, int perms=775) {
	if( std::system(("mkdir -p "+path+" -m "+std::to_string(perms)).c_str()) ) {
		throw std::runtime_error("Failed to create path: "+path);
	}
}
void remove_all(std::string path) {
	if( std::system(("rm -rf "+path).c_str()) ) {
		throw std::runtime_error("Failed to remove all: "+path);
	}
}
void remove_dir(std::string path) {
	if( std::system(("rmdir "+path+" 2> /dev/null").c_str()) ) {
		throw std::runtime_error("Failed to remove dir: "+path);
	}
}
void remove_file(std::string path) {
	if( std::system(("rm -f "+path).c_str()) ) {
		throw std::runtime_error("Failed to remove file: "+path);
	}
}
bool process_exists(pid_t pid) {
	struct stat s;
	return !(stat(("/proc/"+std::to_string(pid)).c_str(), &s) == -1
	         && errno == ENOENT);
}

std::string get_dirname(std::string filename) {
	// TODO: This is crude, but works for our proclog use-case
	return filename.substr(0, filename.find_last_of("/"));
}

class LockFile {
	std::string _lockfile;
	int         _fd;
public:
	LockFile(LockFile const& ) = delete;
	LockFile& operator=(LockFile const& ) = delete;
	LockFile(std::string lockfile) : _lockfile(lockfile) {
		while( true ) {
			_fd = open(_lockfile.c_str(), O_CREAT, 600);
			flock(_fd, LOCK_EX);
			struct stat fd_stat, lockfile_stat;
			fstat(_fd, &fd_stat);
			stat(_lockfile.c_str(), &lockfile_stat);
			// Compare inodes
			if( fd_stat.st_ino == lockfile_stat.st_ino ) {
				// Got the lock
				break;
			}
			close(_fd);
		}
	}
	~LockFile() {
		unlink(_lockfile.c_str());
		flock(_fd, LOCK_UN);
	}
};

class ProcLogMgr {
	static constexpr const char* base_logdir = "/dev/shm/bifrost";
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
					remove_all(std::string(base_logdir) + "/" +
					           std::to_string(pid));
				}
			}
			closedir(dp);
		}
		// Remove the base_logdir if it's empty
		try { remove_dir(base_logdir); }
		catch( std::exception ) {}
	}
	ProcLogMgr()
		: _logdir(std::string(base_logdir) + "/" + std::to_string(getpid())) {
		this->try_base_logdir_cleanup();
		make_dir(base_logdir, 777);
		make_dir(_logdir);
	}
	~ProcLogMgr() {
		try {
			remove_all(_logdir);
			this->try_base_logdir_cleanup();
		} catch( std::exception ) {}
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
		int i = 1;
		while( _logs.count(name) ) {
			// Disambiguate by adding suffix to name
			name = origname + "_" + std::to_string(++i);
		}
		std::string filename = _logdir + "/" + name;
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
