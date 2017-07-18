""" Test a pipeline with repeated FFTs and inverse FFTs """
from timeit import default_timer as timer
import numpy as np
import bifrost as bf
from bifrost import pipeline as bfp
from bifrost import blocks as blocks
from bifrost_benchmarks import PipelineBenchmarker

NUMBER_FFT = 10

class GPUFFTBenchmarker(PipelineBenchmarker):
    """ Test the sigproc read function """
    def run_benchmark(self):
        with bf.Pipeline() as pipeline:
            datafile = "numpy_data0.bin"

            bc = bf.BlockChainer()
            bc.blocks.binary_read(
                    [datafile], gulp_size=32768, gulp_nframe=128, dtype='f32')
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
        data = np.fromfile(file_obj, dtype=np.float32)
    for _ in range(NUMBER_FFT):
        data = np.fft.fft(data)
        data = np.fft.ifft(data)
    end = timer()
    return end-start

t = np.arange(32768*1024)
w = 0.01
s = np.sin(w * 4 * t, dtype='float32')
with open('numpy_data0.bin', 'wb') as myfile: pass
s.tofile('numpy_data0.bin')
gpufftbenchmarker = GPUFFTBenchmarker()
print "Bifrost gets:", gpufftbenchmarker.average_benchmark(4)[0]

print "Regular single-threaded numpy gets:", regular_numpy_fft_pipeline('numpy_data0.bin')
