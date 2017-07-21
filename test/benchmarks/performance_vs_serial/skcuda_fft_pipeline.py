""" Test a pipeline with repeated FFTs and inverse FFTs """
import os
from timeit import default_timer as timer
import numpy as np
from skcuda.fft import fft, Plan, ifft
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

NUMBER_FFT = int(os.environ['NUMBER_FFT'])
SIZE_MULTIPLIER = int(os.environ['SIZE_MULTIPLIER'])
GULP_SIZE = int(os.environ['GULP_SIZE'])

def scikit_gpu_fft_pipeline(filename):
    data = []
    start = timer()
    with open(filename, 'r') as file_obj:
        for _ in range(32768*1024*SIZE_MULTIPLIER//GULP_SIZE):
            data = np.fromfile(file_obj, dtype=np.complex64, count=GULP_SIZE)
            g_data = gpuarray.to_gpu(data)
            plan = Plan(data.shape, np.complex64, np.complex64)
            tmp1 = gpuarray.empty(data.shape, dtype=np.complex64)
            tmp2 = gpuarray.empty(data.shape, dtype=np.complex64)
            fft(g_data, tmp1, plan)
            ifft(tmp1, tmp2, plan)
            for _ in range(NUMBER_FFT-1):
                # Can't do FFT in place for fairness (emulating full pipeline)
                tmp1 = gpuarray.empty(data.shape, dtype=np.complex64)
                fft(tmp2, tmp1, plan)
                tmp2 = gpuarray.empty(data.shape, dtype=np.complex64)
                ifft(tmp1, tmp2, plan)
    end = timer()
    return end-start

print scikit_gpu_fft_pipeline('numpy_data0.bin')

