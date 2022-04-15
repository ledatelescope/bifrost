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

#pragma once

#include <bifrost/config.h>

#include <sys/file.h>  // For flock
#include <sys/stat.h>  // For fstat
#include <sys/types.h> // For getpid
#include <unistd.h>    // For getpid
#include <pwd.h>       // For getpwuid
#include <system_error>

#if defined(__APPLE__) && __APPLE__
#include <sys/sysctl.h>
#endif

std::string get_home_dir(void);
void make_dir(std::string path, int perms=0775);
void remove_files_recursively(std::string path);
void remove_dir(std::string path);
void remove_file(std::string path);
void remove_files_with_suffix(std::string dir, std::string suffix);
bool file_exists(std::string path);
bool process_exists(pid_t pid);

std::string get_dirname(std::string filename);

class LockFile {
	std::string _lockfile;
	int         _fd;
public:
	LockFile(LockFile const& ) = delete;
	LockFile& operator=(LockFile const& ) = delete;
	LockFile(std::string lockfile);
	~LockFile();
};
