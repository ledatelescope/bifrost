from timeit import default_timer as timer
import bifrost as bf
from bifrost import pipeline as bfp
from bifrost import blocks as blocks

with bf.Pipeline() as pipeline:
    fil_file = "../../../data/1chan8bitNoDM.fil"
    data = blocks.read_sigproc([fil_file], gulp_nframe=4096)
    start = timer()
    pipeline.run()
    end = timer()
    print "Wall clock time:", end-start
