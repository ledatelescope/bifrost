#!/bin/bash

# Part 1 - Synthetic data
## Create
python generate_test_data.py
## Use
coverage run --source=bifrost.ring,bifrost,bifrost.pipeline test_file_read_write.py
coverage run --source=bifrost.ring,bifrost,bifrost.pipeline test_fft.py
coverage run --source=bifrost.ring,bifrost,bifrost.pipeline your_first_block.py

# Part 2 - Real data
## Download
python download_breakthrough_listen_data.py -y
## Use
coverage run --source=bifrost.ring,bifrost,bifrost.pipeline test_guppi.py
coverage run --source=bifrost.ring,bifrost,bifrost.pipeline test_guppi_reader.py
coverage run --source=bifrost.ring,bifrost,bifrost.pipeline test_fdmt.py ./testdata/pulsars/blc0_guppi_57407_61054_PSR_J1840%2B5640_0004.fil
