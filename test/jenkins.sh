#!/bin/bash
# This file runs CPU and GPU tests for jenkins
./download_test_data.sh
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
python -c "from bifrost import telemetry; telemetry.disable()"
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
  test_address \
  test_fdmt \
  test_fft \
  test_fir \
  test_guantize \
  test_gunpack \
  test_linalg \
  test_map \
  test_reduce \
  test_romein \
  test_scrunch \
  test_transpose
