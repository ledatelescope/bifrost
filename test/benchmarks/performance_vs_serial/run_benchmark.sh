#!/bin/bash

declare -a fft_array=("1" "2" "3" "4")
declare -a size_array=("2" "4" "8" "16" "32" "64")
declare -a gulp_array=("1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024")

for i in "${fft_array[@]}"
do

for j in "${size_array[@]}"
do

for k in "${gulp_array[@]}"
do

export NUMBER_FFT="$i"
export SIZE_MULTIPLIER="$j"
export GULP_SIZE="$(echo "32768*1024/$k" | bc)"
NUM1="$(python -OO linear_fft_pipeline.py)"
NUM2="$(python skcuda_fft_pipeline.py)"
echo "$NUMBER_FFT, $SIZE_MULTIPLIER, $GULP_SIZE"
echo "Bifrost has $NUM1"
echo "Scikit has $NUM2"
echo "Bifrost is: "
echo "scale=5; $NUM2/$NUM1" | bc
echo "times faster"

done
done
done

#1.23392
#NUMBER_FFT = 4
#SIZE_MULTIPLIER = 32
#GULP_SIZE = 32768*1024//8**2

#1.72889
#export NUMBER_FFT="2"
#export SIZE_MULTIPLIER="32"
#export GULP_SIZE="$(echo '32768*1024/8' | bc)"

# 1.66151
#export NUMBER_FFT="2"
#export SIZE_MULTIPLIER="32"
#export GULP_SIZE="$(echo '32768*1024/16' | bc)"

#1.62643
#export NUMBER_FFT="2"
#export SIZE_MULTIPLIER="32"
#export GULP_SIZE="$(echo '32768*1024/4' | bc)"

# 1.19816
#export NUMBER_FFT="3"
#export SIZE_MULTIPLIER="32"
#export GULP_SIZE="$(echo '32768*1024/8' | bc)"

# 2.40760
#export NUMBER_FFT="1"
#export SIZE_MULTIPLIER="32"
#export GULP_SIZE="$(echo '32768*1024/8' | bc)"

# 2.13087
#export NUMBER_FFT="1"
#export SIZE_MULTIPLIER="32"
#export GULP_SIZE="$(echo '32768*1024/2' | bc)"
