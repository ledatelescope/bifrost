""" Test the sigproc read function """
from timeit import default_timer as timer
import bifrost as bf
from bifrost import pipeline as bfp
from bifrost import blocks as blocks
from bifrost_benchmarks import PipelineBenchmarker

class SigprocBenchmarker(PipelineBenchmarker):
    """ Test the sigproc read function """
    def run_benchmark(self):
        with bf.Pipeline() as pipeline:
            fil_file = "../../../data/1chan8bitNoDM.fil"
            data = blocks.read_sigproc([fil_file], gulp_nframe=4096)
            data.on_data = self.timeit(data.on_data)

            start = timer()
            pipeline.run()
            end = timer()
            self.total_clock_time = end-start

sigproc_benchmarker = SigprocBenchmarker()
print sigproc_benchmarker.average_benchmark(10)
