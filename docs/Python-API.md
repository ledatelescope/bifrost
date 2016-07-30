## Contents

0. [Basic Syntax](#syntax)
1. [Bifrost blocks](#blocks)
4. [[Ring() API]]

## <a name="syntax">Basic Syntax</a>

As described on the [[Home]] page, Bifrost is made up of blocks, rings, and pipelines. Blocks embody *black box processes*, and rings connect these blocks together. A network of blocks and rings is called a pipeline. Bifrost's Python API mirrors these concepts very closely. 

####Let's dive into an example: here we will perform an FFT on an array, and dump the result to a text file.

The following code generates a list of three blocks: a `TestingBlock`, which takes a list of numbers during its initialization (which we give as `[1, 2, 3]`, an `FFTBlock`, which performs a one dimensional FFT on our input, and a `WriteAsciiBlock`, which dumps everything given to it into a text file (which we name as `'logfile.txt'`).


These blocks are created into a sublist, where they are proceeded by a subsublist of input rings, and a subsublist of output rings. The `TestingBlock` gets an output ring which we arbitrarily name `'my array ring'`, the `FFTBlock` gets the same ring for an input ring, and puts its results into an output ring which we name `'fft output ring'` and `WriteAsciiBlock` gets an input ring of the same name. 

This list of blocks, inputs, and outputs, is fed into a `Pipeline` object. Calling `my_pipeline.main()` then initiates the pipeline. 

```python
from bifrost.block import TestingBlock, FFTBlock, WriteAsciiBlock, Pipeline

my_blocks = []
my_blocks.append([TestingBlock([1, 2, 3]), [], ['my array ring']])
my_blocks.append([FFTBlock(), ['my array ring'], ['fft output ring']])
my_blocks.append([WriteAsciiBlock('logfile.txt'), ['fft output ring'], []])
my_pipeline = Pipeline(my_blocks)
my_pipeline.main() #Turn on the pipeline!
```

[POST IMAGE OF GUI EQUIVALENT]

A file named `'logfile.txt'` should now be created and filled with the result of our FFT. 

As you can see, creating a high-throughput Bifrost pipeline from previously written blocks is a trivial process. This is the great thing about Bifrost: once you have modularized your functions into blocks, you can connect them seamlessly into a pipeline, and have your data streamed through in real-time. 

## <a name="pulsarsearch">Create a Pulsar Search Pipeline</a>

[SIMPLE BLOCKS WHICH CALL PRESTO FUNCTIONS]