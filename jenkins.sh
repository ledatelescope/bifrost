# Script for the Jenkins build system
!#/bin/bash
export PYTHONPATH=/data1/mcranmer/usr/lib/python2.7/site-packages
INSTALL_LIB_DIR=$JENKINS_HOME/usr/local/lib INSTALL_INC_DIR=$JENKINS_HOME/usr/local/include NVCC=/usr/local/cuda-6.5/bin/nvcc make -j
