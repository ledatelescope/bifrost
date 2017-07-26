""" Test a pipeline with repeated FFTs and inverse FFTs """
import os
from timeit import default_timer as timer
import numpy as np
import bifrost as bf
from bifrost import pipeline as bfp
from bifrost import blocks as blocks
from bifrost_benchmarks import PipelineBenchmarker

NUMBER_FFT = int(os.environ['NUMBER_FFT'])
SIZE_MULTIPLIER = int(os.environ['SIZE_MULTIPLIER'])
GULP_SIZE = int(os.environ['GULP_SIZE'])
CORES = int(os.environ['BF_CORES'])
assert(CORES > 2)

class GPUFFTBenchmarker(PipelineBenchmarker):
    """ Test the sigproc read function """
    def run_benchmark(self):
        with bf.Pipeline() as pipeline:
            datafile = "numpy_data0.bin"

            bc = bf.BlockChainer()
            bc.blocks.binary_read(
                    [datafile], gulp_size=GULP_SIZE, gulp_nframe=1, dtype='f32', core=CORES-1)
            bc.blocks.copy('cuda', core=CORES-2)
            cur_core = 0
            for _ in range(NUMBER_FFT):
                bc.blocks.fft(['gulped'], axis_labels=['ft_gulped'], core=cur_core)
                cur_core += 1
                cur_core = cur_core % (CORES - 2)
                bc.blocks.fft(['ft_gulped'], axis_labels=['gulped'], inverse=True, core=cur_core)
                cur_core += 1
                cur_core = cur_core % (CORES - 2)

            start = timer()
            pipeline.run()
            end = timer()
            self.total_clock_time = end-start

gpufftbenchmarker = GPUFFTBenchmarker()

t = np.arange(32768*1024*SIZE_MULTIPLIER)
w = 0.01
s = np.sin(w * 4 * t, dtype='float32')
with open('numpy_data0.bin', 'wb') as myfile: pass
s.tofile('numpy_data0.bin')

print gpufftbenchmarker.average_benchmark(1)[0]

