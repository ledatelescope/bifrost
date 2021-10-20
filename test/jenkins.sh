#!/bin/bash
# This file runs CPU and GPU tests for jenkins
./download_test_data.sh
coverage run --source=bifrost.ring,bifrost,bifrost.pipeline -m unittest discover
