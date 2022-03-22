#!/bin/bash
# This file runs CPU and GPU tests for jenkins
./download_test_data.sh
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
python -c "from bifrost import telemetry; telemetry.disable()"
coverage run --source=bifrost.ring,bifrost,bifrost.pipeline -m unittest discover
