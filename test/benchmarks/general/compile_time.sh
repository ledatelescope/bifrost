#!/bin/bash
cd ../../..
time /bin/bash -c "make -Bj && make install"
cd test/benchmarks/general
