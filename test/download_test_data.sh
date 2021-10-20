#!/bin/bash
curl -L -O https://fornax.phys.unm.edu/lwa/data/bf_test_files.tar.gz
tar xzf bf_test_files.tar.gz
mv for_test_suite data
rm bf_test_files.tar.gz

