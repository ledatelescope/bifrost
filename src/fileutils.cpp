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
#include <sstream>

#if defined(HAVE_CXX_FILESYSTEM) && HAVE_CXX_FILESYSTEM
#include <filesystem>
#endif

std::string get_home_dir(void) {
	const char *homedir;
	if ((homedir = getenv("HOME")) == NULL) {
		homedir = getpwuid(getuid())->pw_dir;
	}
	return std::string(homedir);
}

/* NOTE: For convenience on systems with compilers without C++17 support, these
   functions build a shell command and pass it to system(). The PATH argument is
   not shell-quoted or otherwise sanitized, so only use with program constants,
   not with data from command line or config files. */

void make_dir(std::string path, int perms) {
#if defined(HAVE_CXX_FILESYSTEM) && HAVE_CXX_FILESYSTEM
  bool created = std::filesystem::create_directories(path);
  if(created) {
    std::filesystem::permissions(path, (std::filesystem::perms) perms,
                                 std::filesystem::perm_options::replace);
  }
#else
  std::ostringstream cmd;
  cmd << "mkdir -p -m " << std::oct << perms << ' ' << path;
  if( std::system(cmd.str().c_str()) ) {
    throw std::runtime_error("Failed to create path: "+path);
  }
#endif
}
void remove_files_recursively(std::string path) {
#if defined(HAVE_CXX_FILESYSTEM) && HAVE_CXX_FILESYSTEM
  std::filesystem::remove_all(path);
#else
	if( std::system(("rm -rf "+path).c_str()) ) {
		throw std::runtime_error("Failed to remove all: "+path);
	}
#endif
}
void remove_dir(std::string path) {
#if defined(HAVE_CXX_FILESYSTEM) && HAVE_CXX_FILESYSTEM
  std::filesystem::remove(path);
#else
  if(rmdir(path.c_str()) != 0) {
    throw std::runtime_error("Failed to remove dir: "+path);
  }
#endif
}
void remove_file(std::string path) {
#if defined(HAVE_CXX_FILESYSTEM) && HAVE_CXX_FILESYSTEM
	std::filesystem::remove(path);
#else
  if(unlink(path.c_str()) != 0) {
    // Previously this was an 'rm -f', which is silent on non-existent path.
    if(errno != ENOENT) {
      throw std::runtime_error("Failed to remove file: "+path);
    }
  }
#endif
}

#if defined(HAVE_CXX_FILESYSTEM) && HAVE_CXX_FILESYSTEM
static bool ends_with (std::string const &fullString, std::string const &ending) {
#if defined(HAVE_CXX_ENDS_WITH) && HAVE_CXX_ENDS_WITH
  return fullString.ends_with(ending);
#else
// ends_with will be available in C++20; this is suggested as alternative
// at https://stackoverflow.com/questions/874134
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(),
                                    ending.length(), ending));
  } else {
    return false;
  }
#endif
}
#endif

void remove_files_with_suffix(std::string dir, std::string suffix) {
  if(dir.empty()) {
    throw std::runtime_error("Empty DIR argument");
  }
  if(suffix.empty()) {
    throw std::runtime_error("Empty SUFFIX argument");
  }
#if defined(HAVE_CXX_FILESYSTEM) && HAVE_CXX_FILESYSTEM
  // Iterate through the directory's contents and remove the matches
  std::filesystem::path path = dir;
  for(auto const& entry : std::filesystem::directory_iterator{dir}) {
    std::filesystem::path epath = entry.path();
    if( ends_with(epath.string(), suffix) ) {
      std::filesystem::remove(epath);
    }
  }
#else
  std::string wild = dir + "/*" + suffix;
  if( std::system(("rm -f "+wild).c_str()) ) {
    throw std::runtime_error("Failed to remove files: "+wild);
  }
#endif
}

bool file_exists(std::string path) {
#if defined(HAVE_CXX_FILESYSTEM) && HAVE_CXX_FILESYSTEM
  return std::filesystem::exists(path);
#else
    struct stat s;
    return !(stat(path.c_str(), &s) == -1
	         && errno == ENOENT);
#endif
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
  return file_exists("/proc/"+std::to_string(pid));
#endif
}

std::string get_dirname(std::string filename) {
#if defined(HAVE_CXX_FILESYSTEM) && HAVE_CXX_FILESYSTEM
  std::filesystem::path path = filename;
	return (path.parent_path()).string();
#else
	// TODO: This is crude, but works for our proclog use-case
	return filename.substr(0, filename.find_last_of("/"));
#endif
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
