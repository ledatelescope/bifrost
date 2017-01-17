#!/bin/bash
# This file runs CPU-safe tests for travis-ci
./download_test_data.sh
python -m unittest test_block &&
python -m unittest test_sigproc &&
python -m unittest test_resizing 
