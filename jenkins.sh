#!/bin/bash
# Script for the Jenkins build system

while [ "$1" != "" ]; do 
    case $1 in
        --build )
            export INSTALL_LIB_DIR=$JENKINS_HOME/usr/local/lib
            export INSTALL_INC_DIR=$JENKINS_HOME/usr/local/include
            NVCC=/usr/local/cuda-6.5/bin/nvcc make -j
            make install
            cd python
            python setup.py install --prefix=$JENKINS_HOME/usr/local
            ;;
        --test )
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JENKINS_HOME/usr/local/lib:/usr/local/cuda-6.5/targets/x86_64-linux/lib:/usr/local/cuda-6.5/targets/x86_64-linux/lib
            export LIBRARY_PATH=$LIBRARY_PATH:$JENKINS_HOME/usr/local/include
            export PYTHONPATH=$JENKINS_HOME/usr/local/lib/python2.7:$PYTHONPATH:$JENKINS_HOME/usr/local/lib/python2.7/site-packages:$JENKINS_HOME/usr/local/lib/python2.7/site-packages/bifrost;
            cd test;
            python -m unittest discover
            ;;
        * )
            exit 1
    esac
    shift
done
