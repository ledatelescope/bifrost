"""
# your_first_block.py

This testbench initializes a simple bifrost pipeline that reads from a binary file,
and then writes the data to an output file. 
"""
import os
import numpy as np
import bifrost.pipeline as bfp
from bifrost.blocks import BinaryFileReadBlock, BinaryFileWriteBlock
import glob
from datetime import datetime
from copy import deepcopy
from pprint import pprint

class UselessAddBlock(bfp.TransformBlock):
    def __init__(self, iring, n_to_add, *args, **kwargs):
        super(UselessAddBlock, self).__init__(iring, *args, **kwargs)
        self.n_to_add = n_to_add

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        ohdr["name"] += "_with_added_value"
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        idata = ispan.data + self.n_to_add
        odata = ospan.data

        odata[...] = idata
        return out_nframe

class PrintStuffBlock(bfp.SinkBlock):
    def __init__(self, iring, *args, **kwargs):
        super(PrintStuffBlock, self).__init__(iring, *args, **kwargs)
        self.n_iter = 0

    def on_sequence(self, iseq):
        print("[%s]" % datetime.now())
        print(iseq.name)
        pprint(iseq.header)
        self.n_iter = 0

    def on_data(self, ispan):
        now = datetime.now()
        if self.n_iter % 100 == 0:
            print("[%s] %s" % (now, ispan.data))
        self.n_iter += 1


if __name__ == "__main__":

    # Setup pipeline
    filenames   = sorted(glob.glob('testdata/sin_data*.bin'))

    b_read      = BinaryFileReadBlock(filenames, 32768, 1, 'f32')
    b_add       = UselessAddBlock(b_read, n_to_add=100)
    b_print     = PrintStuffBlock(b_read)
    b_print2    = PrintStuffBlock(b_add)

    # Run pipeline
    pipeline = bfp.get_default_pipeline()
    print pipeline.dot_graph()
    pipeline.run()
