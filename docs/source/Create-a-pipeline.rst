Create a Pipeline
=================

In this tutorial, we will create a simple
pipeline and execute it. Later on, we'll create
our own version of a block and use it 
in the pipeline.

With Bifrost, there is one main module you will
be calling as a user: ``bifrost.pipeline``. This
handles all the behind-the-scenes pipeline construction,
giving you a high-level view at arranging a series of
blocks. 

We would like to construct the following pipeline, 
which will serve to calculate the beats per minute
of a song. As we will soon see, some intermediate
operations will be required to get the bpm, and
we can then write our own block.

1. Read in a ``.wav`` file to a ring buffer.
#. Channelize it with a GPU FFT.
#. Write it back to disk as a filterbank file.

This will require bifrost blocks which:

1. Read in the ``.wav`` file.
#. Copy the raw data to the GPU.
#. Split the time axis into chunks which we can FFT over.
#. FFT this new axis.
#. Take the modulus squared of these FFTs.
#. Transpose this data into a format compatible with the sigproc writer.
#. Copy the data back to the CPU.
#. Convert the data into integer data types.
#. Write this data to a filterbank file.

This file could then be used to do things like calculating
the beats per minute of the song at different points of time, or
could be used to just view the frequency components of the song with time.

First, ensure you have a working Bifrost installation. You should
also have some CUDA-compatible GPUs to run this example.

Now, let's create the pipeline.

The first thing to do is to actually load in Bifrost. Load in the base
library as ``bf``:

.. code:: python

    import bifrost as bf

Next, let's load in some function libraries. We want ``blocks``,
which is the block module in Bifrost, which is a collection of 
previously-written blocks for various functionality,and
``views``, which is a library for manipulations of ring headers.

.. code:: python

    import bifrost.blocks as blocks
    import bifrost.views as views

