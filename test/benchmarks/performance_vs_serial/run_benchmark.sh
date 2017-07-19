#!/bin/bash

export NUMBER_FFT="2"
export SIZE_MULTIPLIER="32"
export GULP_SIZE="$(echo '32768*1024' | bc)"
NUM1="$(python -OO test1.py)"
NUM2="$(python test2.py)"
echo "Bifrost has $NUM1"
echo "Scikit has $NUM2"
echo "Bifrost is: "
echo "scale=5; $NUM2/$NUM1" | bc
echo "times faster"

#1.23392
#NUMBER_FFT = 4
#SIZE_MULTIPLIER = 32
#GULP_SIZE = 32768*1024//8**2

#1.72889
#export NUMBER_FFT="2"
#export SIZE_MULTIPLIER="32"
#export GULP_SIZE="$(echo '32768*1024/8' | bc)"
