#!/bin/bash

ROOT_PATH=`dirname $0`
cd ${ROOT_PATH}/../bifrost
INSTALL_PREFIX=`python -c "import os,sys; print(os.path.dirname(os.path.dirname(sys.executable)))"`
./configure --prefix=${INSTALL_PREFIX}
make -j all
make install
