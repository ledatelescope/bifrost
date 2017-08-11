#!/bin/bash
# This file runs CPU-safe tests for travis-ci
./download_test_data.sh
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
coverage run --source=bifrost.ring,bifrost,bifrost.pipeline -m unittest \
  test_block \
  test_sigproc \
  test_resizing \
  test_quantize \
  test_unpack \
  test_print_header \
  test_pipeline_cpu \
  test_serialize \
  test_binary_io \
  test_address
