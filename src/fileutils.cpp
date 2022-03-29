/*
 * Copyright (c) 2019-2022, The Bifrost Authors. All rights reserved.
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

#include "fileutils.hpp"

std::string get_home_dir(void) {
	const char *homedir;
	if ((homedir = getenv("HOME")) == NULL) {
		homedir = getpwuid(getuid())->pw_dir;
	}
	return std::string(homedir);
}

/* NOTE: For convenience, these functions build a shell command and pass it to
   system(). The PATH argument is not shell-quoted or otherwise sanitized, so
   only use with program constants, not with data from command line or config
   files. Might eventually implement these with C++/boost filesystem library. */

void make_dir(std::string path, int perms) {
	if( std::system(("mkdir -p -m "+std::to_string(perms)+" "+path).c_str()) ) {
		throw std::runtime_error("Failed to create path: "+path);
	}
}
void remove_files_recursively(std::string path) {
	if( std::system(("rm -rf "+path).c_str()) ) {
		throw std::runtime_error("Failed to remove all: "+path);
	}
}
void remove_dir(std::string path) {
	if( std::system(("rmdir "+path+" 2> /dev/null").c_str()) ) {
		throw std::runtime_error("Failed to remove dir: "+path);
	}
}
void remove_file_glob(std::string path) {
  // Often, PATH contains wildcard, so this can't just be unlink system call.
	if( std::system(("rm -f "+path).c_str()) ) {
		throw std::runtime_error("Failed to remove file: "+path);
	}
}

bool file_exists(std::string path) {
    struct stat s;
    return !(stat(path.c_str(), &s) == -1
	         && errno == ENOENT);
}
bool process_exists(pid_t pid) {
#if defined(__APPLE__) && __APPLE__

  // Based on information from:
	//   https://developer.apple.com/library/archive/qa/qa2001/qa1123.html
	
  static const int name[] = { CTL_KERN, KERN_PROC, KERN_PROC_ALL, 0 };
	kinfo_proc *proclist = NULL;
	int err, found = 0;
	size_t len, count;
	len = 0;
	err = sysctl((int *) name, (sizeof(name) / sizeof(*name)) - 1,
               NULL, &len, NULL, 0);
	if( err == 0 ) {
		proclist = (kinfo_proc*) ::malloc(len);
		err = sysctl((int *) name, (sizeof(name) / sizeof(*name)) - 1,
                 proclist, &len, NULL, 0);
		if( err == 0 ) {
			count = len / sizeof(kinfo_proc);
			for(int i=0; i<count; i++) {
				pid_t c_pid = proclist[i].kp_proc.p_pid;
				if( c_pid == pid ) {
					found = 1;
					break;
				}
			}
		}
		::free(proclist);
	}
	return (bool) found;
#else
	struct stat s;
	return !(stat(("/proc/"+std::to_string(pid)).c_str(), &s) == -1
	         && errno == ENOENT);
#endif
}

std::string get_dirname(std::string filename) {
	// TODO: This is crude, but works for our proclog use-case
	return filename.substr(0, filename.find_last_of("/"));
}

/* NOTE: In case of abnormal exit (such as segmentation fault or other signal),
   the lock file will not be removed, and the next attempt to lock might busy-
   wait until the file is manually deleted. If this is a common issue, we could
   potentially write the PID into the lock file to help with tracking whether
   the process died. */
LockFile::LockFile(std::string lockfile) : _lockfile(lockfile) {
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

LockFile::~LockFile() {
	unlink(_lockfile.c_str());
	flock(_fd, LOCK_UN);
}
