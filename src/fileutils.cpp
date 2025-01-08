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
#include <iostream>
#include <cstring> // strerror

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

LockFile::LockFile(std::string lockfile) : _lockfile(lockfile) {
  time_t start = time(NULL), elapsed = 0;
  long busycount = 0;
  long busyinterval = 1 << 17;
  pid_t pid = getpid();
  while( true ) {
    _fd = open(_lockfile.c_str(), O_CREAT|O_RDWR, S_IRUSR|S_IWUSR);
    if( _fd == -1 ) {
      if(busycount == 0) {
        std::cerr << "ERROR: could not create " << lockfile << ": "
                  << strerror(errno) << std::endl;
      }
    }
    else {
      if( flock(_fd, LOCK_EX | LOCK_NB) == 0 ) {
        struct stat fd_stat, lockfile_stat;
        fstat(_fd, &fd_stat);
        stat(_lockfile.c_str(), &lockfile_stat);
        if( fd_stat.st_ino == lockfile_stat.st_ino ) {
          // We acquired the lock. Announce it only if we announced waiting.
          if(elapsed > 0) {
            std::cerr << "NOTE: acquired " << lockfile << std::endl;
          }
          // Exit the busy loop
          break;
        }
        // Locking succeeded, but inodes were different so try again.
        flock(_fd, LOCK_UN);
      }
      close(_fd);
    }
    busycount++;
    if(busycount % busyinterval == 0) {
      elapsed = time(NULL) - start;
      if(elapsed >= 5) {
        std::cerr << "NOTE: waiting " << elapsed
                  << "s for " << lockfile << std::endl;
        busyinterval = busycount;
      }
    }
  }
  // We have the lock, so try writing PID.
  if(ftruncate(_fd, 0) == -1) {
    std::cerr << "WARNING: could not truncate " << lockfile << ": "
              << strerror(errno) << std::endl;
  }
  else {
    std::ostringstream ss;
    ss << pid << '\n';
    std::string buf = ss.str();
    if(write(_fd, buf.c_str(), buf.size()) != (ssize_t)buf.size()) {
      std::cerr << "WARNING: could not write to " << lockfile << ": "
                << strerror(errno) << std::endl;
    }
  }
}

LockFile::~LockFile() {
  unlink(_lockfile.c_str());
  flock(_fd, LOCK_UN);
  close(_fd);
}


#ifdef FILEUTILS_LOCKFILE_TEST
// This is a little test program for LockFile. Run it in two terminals
// to simulate race conditions, or kill one to leave a a leftover lock file.
// Output should look something like this:

//    Initiating lock...
//    NOTE: waiting 5s for ./myfile.lock
//    NOTE: waiting 9s for ./myfile.lock
//    NOTE: waiting 17s for ./myfile.lock
//    NOTE: waiting 34s for ./myfile.lock
//    [Delete the file from another terminal]
//    NOTE: acquired ./myfile.lock
//    Lock acquired... exiting

int main() {
  std::cerr << "Initiating lock..." << std::endl;
  LockFile lock("./myfile.lock");
  std::cerr << "Lock acquired, 'working' for 15s..." << std::endl;
  sleep(15);
  std::cerr << "Work finished, releasing lock..." << std::endl;
  return 0;
}
#endif
