import numpy as np
from timeit import default_timer as timer
import bifrost as bf
from bifrost import pipeline as bfp
from bifrost import blocks as blocks

class Benchmarker(object):
    def __init__(self):
        self.total_clock_time = 0
        self.relevant_clock_time = 0

    def timeit(self, method):
        """ Decorator for timing execution of a method """
        def timed(*args, **kw):
            ts = timer()
            result = method(*args, **kw)
            te = timer()

            self.relevant_clock_time += te-ts
            return result
        return timed
    
    def reset_times(self):
        self.total_clock_time = 0
        self.relevant_clock_time = 0

    def run_benchmark(self):
        with bf.Pipeline() as pipeline:
            fil_file = "../../../data/1chan8bitNoDM.fil"
            data = blocks.read_sigproc([fil_file], gulp_nframe=4096)
            data.on_data = self.timeit(data.on_data)

            start = timer()
            pipeline.run()
            end = timer()
            self.total_clock_time = end-start

    def average_benchmark(self, number_runs):
        """ First test is always longer """
        self.run_benchmark()
        self.reset_times()

        total_clock_times = np.zeros(number_runs)
        relevant_clock_times = np.zeros(number_runs)
        for i in range(number_runs):
            self.run_benchmark()
            total_clock_times[i] = self.total_clock_time
            relevant_clock_times[i] = self.relevant_clock_time
            self.reset_times()
        return np.average(total_clock_times), np.average(relevant_clock_times)

sigproc_benchmarker = Benchmarker()
print sigproc_benchmarker.average_benchmark(10)
