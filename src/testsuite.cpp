/*                                               -*- indent-tabs-mode:nil -*-
 * Copyright (c) 2022, The Bifrost Authors. All rights reserved.
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

#include <bifrost/testsuite.h>
#include <bifrost/config.h>
#include <fileutils.hpp>

#include <sys/types.h> // For getpid
#include <unistd.h>    // For getpid
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

using namespace std;

// General output format, used for tracing and error reporting.
#define TPRINT(level, arg)                \
  (cout << "testsuite: " << level << ": " \
   << __FUNCTION__ << ": " << arg << " @" << __LINE__ << '\n')

// Tracing for debug builds only.
#if BF_DEBUG_ENABLED
#define TDEBUG(arg) TPRINT("DEBUG", arg)
#else
#define TDEBUG(arg) {}
#endif

// Final result of a test: use the inversion like exit() where 0 means success,
// so we can add to count the failures.
typedef enum { OK = 0, FAIL = 1 } TestResult;

ostream& operator << (ostream& out, TestResult r) {
  return out << (r == OK? "ok" : "FAIL");
}

// Assertions output on failure, whether or not debug build.
#define ASSERT(pred, mesg) \
  if(!(pred)) { TPRINT("ERROR", mesg); return FAIL; }

#define ASSERT_PATH_EXISTS(path) \
  ASSERT(file_exists(path), "Path does not exist: " << path)

#define ASSERT_PATH_NOT_EXISTS(path) \
  ASSERT(!file_exists(path), "Path exists: " << path)

// Create a temporary directory for any file tests, then clean it up after.
struct TestDir {
  char* path = NULL;
  TestDir();
  ~TestDir();
  const string mkFileName(const string& baseName);
  const string mkFile(const string& baseName);
};

TestDir::TestDir() {
  // Find a temporary directory, either $TMPDIR or /tmp
  const char *tmpdir = getenv("TMPDIR");
  if(tmpdir == NULL) {
    tmpdir = "/tmp";
  }
  // Buffer that mkdtemp can modify; it cannot use a string constant.
  unsigned tmpdirLen = strlen(tmpdir);
  path = new char [tmpdirLen + 40];
  strcpy(path, tmpdir);
  // Add a slash if needed (TMPDIR on Mac may already end in slash?)
  if(tmpdir[tmpdirLen-1] != '/') {
    strcat(path, "/");
  }
  strcat(path, "bifrost-testsuite.XXXXXX");
  if(mkdtemp(path) == NULL) {
    throw runtime_error(string("mkdtemp failure creating ") + path);
  }
  TDEBUG("created " << path);
}

TestDir::~TestDir() {
  TDEBUG("removing " << path);
  remove_files_recursively(path);
  delete [] path;
}

const string TestDir::mkFileName(const string& baseName) {
  return string(path) + '/' + baseName;
}

const string TestDir::mkFile(const string& baseName) {
  string fname = mkFileName(baseName);
  ofstream file (fname);
  file.close();
  return fname;
}

static TestResult test_this_process_exists() {
  pid_t pid = getpid();
  ASSERT(process_exists(pid), "my PID " << pid << " not found.");
  return OK;
}

static TestResult test_dir() {
  string path;
  { TestDir tmp;
    path = tmp.path;
    ASSERT_PATH_EXISTS(path);
  }
  ASSERT_PATH_NOT_EXISTS(path);
  return OK;
}

static TestResult test_make_then_remove_dir() {
  TestDir tmp;
  string dir = tmp.mkFileName("w0w.d");
  TDEBUG("make_dir " << dir);
  make_dir(dir);
  ASSERT_PATH_EXISTS(dir);

  TDEBUG("remove_dir " << dir);
  remove_dir(dir);
  ASSERT_PATH_NOT_EXISTS(dir);

  return OK;
}

static TestResult test_create_then_remove_file() {
  TestDir tmp;
  string fname = tmp.mkFile("rain.b0w");
  TDEBUG("mkFile " << fname);
  ASSERT_PATH_EXISTS(fname);

  TDEBUG("remove_file " << fname);
  remove_file(fname);
  ASSERT_PATH_NOT_EXISTS(fname);

  return OK;
}

static TestResult test_remove_files_by_extension() {
  TestDir tmp;
  // We'll glob-delete the .bak files and leave the rest.
  vector<string> names =
  { "cheez.bak", "Floop.3.bak",
      "cheez.txt", "zan.tex", "bobak", "lib.baks"
      };

  for(unsigned i = 0; i < names.size(); i++) {
    names.at(i) = tmp.mkFile(names.at(i));
    TDEBUG("mkFile " << names.at(i));
    ASSERT_PATH_EXISTS(names.at(i));
  }

  TDEBUG("removing *.bak");
  remove_files_with_suffix(tmp.path, ".bak");

  ASSERT_PATH_NOT_EXISTS(names.at(0));
  ASSERT_PATH_NOT_EXISTS(names.at(1));

  ASSERT_PATH_EXISTS(names.at(2));
  ASSERT_PATH_EXISTS(names.at(3));
  ASSERT_PATH_EXISTS(names.at(4));
  ASSERT_PATH_EXISTS(names.at(5));
  return OK;
}

int bfTestSuite() {
  int numFails = 0;

  numFails += test_this_process_exists();
  numFails += test_dir();
  numFails += test_make_then_remove_dir();
  numFails += test_create_then_remove_file();
  numFails += test_remove_files_by_extension();

  switch(numFails) {
  case 0: TDEBUG("success"); break;
  case 1: TDEBUG("1 failure"); break;
  default: TDEBUG(numFails << " failures");
  }
  return numFails;
}
