# Bifrost
[![Build status](http://mcranmer.com/access/status.svg)](http://mcranmer.com/access/status.txt)

A stream processing framework for high-throughput applications.

## Feature overview

 * Designed for sustained high-throughput stream processing
 * Python and C++ APIs wrap fast C++/CUDA backend
 * Native support for both system (CPU) and CUDA (GPU) memory spaces and computation

 * Main modules
  - Ring buffer: Flexible and thread safe, supports CPU and GPU memory spaces
  - Transpose: Arbitrary transpose function for ND arrays

 * Experimental modules
  - UDP: Fast data capture with memory reordering and unpacking
  - Radio astronomy: High-performance signal processing operations

## Installation

Edit user.mk to suit your system, then run:

    $ make -j
    $ sudo make install

which will install the library and headers into /usr/local/lib and
/usr/local/include respectively.

## Python interface

Install dependencies:

 * [PyCLibrary fork](https://github.com/MatthieuDartiailh/pyclibrary)

Install bifrost module:

    $ cd python/
    $ sudo python setup.py install

Note that the bifrost module's use of PyCLibrary means it must have
access to both the bifrost shared library and the bifrost headers at
import time. The LD_LIBRARY_PATH and BIFROST_INCLUDE_PATH environment
variables can be used to add search paths for these dependencies
respectively.

## Contributors

 * Ben Barsdell
 * Daniel Price
 * Miles Cranmer
 * Hugh Garsden
