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

#pragma once

#include <sys/file.h>  // For flock
#include <sys/stat.h>  // For fstat
#include <sys/types.h> // For getpid
#include <unistd.h>    // For getpid
#include <system_error>

inline void make_dir(std::string path, int perms=775) {
	if( std::system(("mkdir -p "+path+" -m "+std::to_string(perms)).c_str()) ) {
		throw std::runtime_error("Failed to create path: "+path);
	}
}
inline void remove_all(std::string path) {
	if( std::system(("rm -rf "+path).c_str()) ) {
		throw std::runtime_error("Failed to remove all: "+path);
	}
}
inline void remove_dir(std::string path) {
	if( std::system(("rmdir "+path+" 2> /dev/null").c_str()) ) {
		throw std::runtime_error("Failed to remove dir: "+path);
	}
}
inline void remove_file(std::string path) {
	if( std::system(("rm -f "+path).c_str()) ) {
		throw std::runtime_error("Failed to remove file: "+path);
	}
}
inline bool file_exists(std::string path) {
    struct stat s;
    return !(stat(path.c_str(), &s) == -1
	         && errno == ENOENT);
}
inline bool process_exists(pid_t pid) {
	struct stat s;
	return !(stat(("/proc/"+std::to_string(pid)).c_str(), &s) == -1
	         && errno == ENOENT);
}

inline std::string get_dirname(std::string filename) {
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

