""" Test a pipeline with repeated FFTs and inverse FFTs """
from timeit import default_timer as timer
import numpy as np
import bifrost as bf
from bifrost import pipeline as bfp
from bifrost import blocks as blocks
from bifrost_benchmarks import PipelineBenchmarker
from scipy import fftpack
from skcuda.fft import fft, Plan, ifft
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

NUMBER_FFT = 5
SIZE_MULTIPLIER=2*2*2

class GPUFFTBenchmarker(PipelineBenchmarker):
    """ Test the sigproc read function """
    def run_benchmark(self):
        with bf.Pipeline() as pipeline:
            datafile = "numpy_data0.bin"

            bc = bf.BlockChainer()
            bc.blocks.binary_read(
                    [datafile], gulp_size=32768*1024//8, gulp_nframe=1, dtype='f32')
            bc.blocks.copy('cuda')
            for _ in range(NUMBER_FFT):
                bc.blocks.fft(['gulped'], axis_labels=['ft_gulped'])
                bc.blocks.fft(['ft_gulped'], axis_labels=['gulped'], inverse=True)

            start = timer()
            pipeline.run()
            end = timer()
            self.total_clock_time = end-start

def regular_numpy_fft_pipeline(filename):
    data = []
    start = timer()
    with open(filename, 'r') as file_obj:
        data = np.fromfile(file_obj, dtype=np.float32).astype(np.complex64)
    for _ in range(NUMBER_FFT):
        data = np.fft.fft(data)
        data = np.fft.ifft(data)
    end = timer()
    return end-start

def scipy_fftpack_fft_pipeline(filename):
    data = []
    start = timer()
    with open(filename, 'r') as file_obj:
        data = np.fromfile(file_obj, dtype=np.float32).astype(np.complex64)
    for _ in range(NUMBER_FFT):
        data = fftpack.fft(data)
        data = fftpack.ifft(data)
    end = timer()
    return end-start

def scikit_gpu_fft_pipeline(filename):
    data = []
    start = timer()
    with open(filename, 'r') as file_obj:
        for _ in range(8*SIZE_MULTIPLIER):
            data = np.fromfile(file_obj, dtype=np.float32, count=32768*1024//8).astype(np.complex64)
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

t = np.arange(32768*1024*SIZE_MULTIPLIER)
w = 0.01
s = np.sin(w * 4 * t, dtype='float32')
with open('numpy_data0.bin', 'wb') as myfile: pass
s.tofile('numpy_data0.bin')

gpufftbenchmarker = GPUFFTBenchmarker()

print "Bifrost gets:", gpufftbenchmarker.average_benchmark(2)[0]

#print "Regular single-threaded numpy gets:", regular_numpy_fft_pipeline('numpy_data0.bin')

#print "scipy fftpack gets:", scipy_fftpack_fft_pipeline('numpy_data0.bin')

print "scikit fftpack gets:", scikit_gpu_fft_pipeline('numpy_data0.bin')

