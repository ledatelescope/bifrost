#!/bin/bash
./download_test_data.sh
python -m unittest test_block
python -m unittest test_sigproc
