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

sigproc_benchmarker = Benchmarker()
sigproc_benchmarker.run_benchmark()
print sigproc_benchmarker.total_clock_time,
print sigproc_benchmarker.relevant_clock_time
