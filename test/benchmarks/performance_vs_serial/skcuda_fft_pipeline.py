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
GULP_FRAME_FFT = int(os.environ['GULP_FRAME_FFT'])

# This needs to be here because gulps are now technically twice
# as large
COMPLEX_MULTIPLIER = 2

def scikit_gpu_fft_pipeline(filename):
    data = []
    start = timer()
    with open(filename, 'r') as file_obj:
        for _ in range(((32768*1024*SIZE_MULTIPLIER//GULP_SIZE)//COMPLEX_MULTIPLIER)//GULP_FRAME_FFT):
            data = np.fromfile(file_obj, dtype=np.complex64, count=GULP_SIZE*GULP_FRAME_FFT).reshape((GULP_FRAME_FFT, GULP_SIZE))
            g_data = gpuarray.to_gpu(data)
            plan = Plan(data.shape[1], np.complex64, np.complex64, batch=GULP_FRAME_FFT)
            plan_inverse = Plan(data.shape[1], np.complex64, np.complex64, batch=GULP_FRAME_FFT)
            tmp1 = gpuarray.empty(data.shape, dtype=np.complex64)
            tmp2 = gpuarray.empty(data.shape, dtype=np.complex64)
            fft(g_data, tmp1, plan)
            ifft(tmp1, tmp2, plan_inverse)
            for _ in range(NUMBER_FFT-1):
                # Can't do FFT in place for fairness (emulating full pipeline)
                tmp1 = gpuarray.empty(data.shape, dtype=np.complex64)
                fft(tmp2, tmp1, plan)
                tmp2 = gpuarray.empty(data.shape, dtype=np.complex64)
                ifft(tmp1, tmp2, plan_inverse)
    end = timer()
    return end-start

print scikit_gpu_fft_pipeline('numpy_data0.bin')

