#!/bin/bash
curl -L -O http://mcranmer.com/data/bf_test_files.tar.gz
tar xzf bf_test_files.tar.gz
mv for_test_suite data
rm bf_test_files.tar.gz

