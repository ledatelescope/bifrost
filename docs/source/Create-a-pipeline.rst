Create a Pipeline
=================

In this tutorial, we will create a simple
pipeline and execute it. Later on, we'll create
our own version of a block and use it 
in the pipeline.

With Bifrost, there is one main module you will
be calling as a user: `bifrost.pipeline`. This
handles all the behind-the-scenes pipeline construction,
giving you a high-level view at arranging a series of
blocks. 

We would like to construct the following pipeline, 
which will serve to calculate the beats per minute
of a song. As we will soon see, some intermediate
operations will be required to get the bpm, and
we can then write our own block.

1. Read in a `.wav` file to a ring buffer.
#. Perform a downsampling.
