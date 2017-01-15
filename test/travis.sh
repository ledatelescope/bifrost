#!/bin/bash
cd /bifrost/python/test
./download_test_data.sh
python -m unittest discover
