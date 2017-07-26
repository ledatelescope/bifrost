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
GULP_FRAME = int(os.environ['GULP_FRAME'])
GULP_FRAME_FFT = int(os.environ['GULP_FRAME_FFT'])

class GPUFFTBenchmarker(PipelineBenchmarker):
    """ Test the sigproc read function """
    def run_benchmark(self):
        with bf.Pipeline() as pipeline:
            datafile = "numpy_data0.bin"

            bc = bf.BlockChainer()
            bc.blocks.binary_read(
                    [datafile], gulp_size=GULP_SIZE, gulp_nframe=GULP_FRAME, dtype='cf32')
            bc.blocks.copy('cuda', gulp_nframe=GULP_FRAME)
            for _ in range(NUMBER_FFT):
                bc.blocks.fft(['gulped'], axis_labels=['ft_gulped'], gulp_nframe=GULP_FRAME_FFT)
                bc.blocks.fft(['ft_gulped'], axis_labels=['gulped'], inverse=True, gulp_nframe=GULP_FRAME_FFT)

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
