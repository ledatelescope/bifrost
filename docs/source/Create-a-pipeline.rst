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

Now, let's create our data "source," our source block. This is the
block that feeds our pipeline with data. In this example,
we work with a ``.wav`` file. I assume that you have own some
sort of audio file and can convert it, using, e.g.,
`online-convert <http://audio.online-convert.com/convert-to-wav>`_.

Now, I want to load this into Bifrost. The syntax for
this instance is:

.. code:: python

    raw_data = blocks.read_wav(['heyjude.wav'], gulp_nframe=4096)

Where ``['heyjude.wav']`` is a list of ``.wav`` files, which is in this
case, a sample of `Hey Jude`. ``gulp_nframe`` is an argument passed
to this block which tells it how many `frames` of data to `gulp` at once.

Some terminology:

- `frame`: One chunk of data. In this case, it is a single sample of the
  audio file. By setting ``gulp_nframe=4096``, we tell the block to read
  in 4096 samples at a time, and put these into the ring buffer at once.
- `gulp`: One read or write of the ring buffer. Imagine the block
  taking a gulp of data. Then ``gulp_nframe`` is how many frames are
  in that gulp.


Now, ``raw_data`` is now a reference to a ``block`` object, which implicitly
points at the `ring buffer` which will hold the raw ``.wav`` data.

Next, we want to put this data onto the GPU. Bifrost makes this simple.
Insert a copy block as follows:

.. code:: python

    gpu_raw_data = blocks.copy(raw_data, space='cuda')

In this line we are telling Bifrost to create a new block, a ``copy`` block,
and set its input to be the ``raw_data`` variable which is the source block
for our audio file. Then, by setting ``space='cuda'``, we tell Bifrost
to create a ring in GPU memory, and copy all of the contents of ``raw_data``
into this new ring. With this GPU ring, we can connect more blocks and
do GPU processing.

Now, since we only want to do a Fourier transform at different parts of the
song, not the entirety of the song, we want to chunk up this audio file
into segments over which we can Fourier transform. This lets us get a
frequency view at various points of the song. Since our data comes
as one long time stream, we want to break it up into parts. Bifrost lets
you do this without extra processing. You simply manipulate the `header`
of the ring, which stores all of the descriptions for the ring. These
manipulations are accomplished with ``views``:

.. code:: python

    chunked_data = views.split_axis(gpu_raw_data, 'time', 256, label='fine_time')

What have we done here? We took ``gpu_raw_data``, which is a block on the GPU,
and which implicitly points to its output ring buffer which sits on the GPU,
and put it into the ``split_axis`` view. We said take the ``'time'`` axis
of this ring, and break it up into ``256``-size chunks. Create a new
axis for this data, and call that axis ``'fine_time'``.

Note that `views` are special in that they do not actually modify the data.
They just modify the metadata, which lets blocks interpret the data
differently.
