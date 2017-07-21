#!/bin/bash

declare -a fft_array=("1" "2")
declare -a size_array=("2" "4" "8" "16")
declare -a gulp_array=("1" "2" "4" "8" "16" "32" "64" "128")

echo "start key, # FFT's/2, size multiplier, gulp size, bifrost execution time, skcuda execution time, speedup, end key"

for i in "${fft_array[@]}"
do

for j in "${size_array[@]}"
do

for k in "${gulp_array[@]}"
do

echo -n ">>>START, "
export NUMBER_FFT="$i"
export SIZE_MULTIPLIER="$j"
export GULP_SIZE="$(echo "32768*1024/$k" | bc)"
echo -n "$NUMBER_FFT, $SIZE_MULTIPLIER, $GULP_SIZE, "
NUM1="$(python -OO linear_fft_pipeline.py)"
NUM2="$(python skcuda_fft_pipeline.py)"
speedup=$(echo "scale=5; $NUM2/$NUM1" | bc)
echo "$NUM1, $NUM2, $speedup, END<<<"

done
done
done
