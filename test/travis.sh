#!/bin/bash
# This file runs CPU-safe tests for travis-ci
./download_test_data.sh
python -m unittest test_block test_sigproc test_resizing test_quantize test_unpack test_print_header
