#!/bin/bash

# Part 1 - Synthetic data
## Create
python generate_test_data.py
## Use
python test_file_read_write.py
python test_fft.py
python your_first_block.py

# Part 2 - Real data
## Download
python download_breakthrough_listen_data.py -y
## Use
python test_guppi.py
python test_guppi_reader.py
python test_fdmt.py ./testdata/pulsars/blc0_guppi_57407_61054_PSR_J1840%2B5640_0004.fil
