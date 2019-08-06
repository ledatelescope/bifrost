Introduction to Bifrost
=======================

Bifrost is ...
--------------

A stream processing framework, created to ease the development of
high-throughput processing CPU/GPU pipelines. It is specifically
designed for digital signal processing (DSP) applications within radio
astronomy. A portable C API is provided, along with C++ and Python
wrappers.

The heart of bifrost is a flexible *ring* buffer implementation that
allows different signal processing *blocks* to be connected to form a
*pipeline*. Each block may be assigned to a CPU core, and the ring
buffers are used to transport data to and from blocks. Processing blocks
may be run on either the CPU or GPU, and the ring buffer will take care
of memory copies between the CPU and GPU spaces.

Core concepts
-------------

Streaming data
^^^^^^^^^^^^^^

The purpose of bifrost is to allow rapid development of *streaming* DSP
pipelines; that is, it is designed for *stream-like* data. A simple
example of a data stream is the time series voltage data from a radio
telescope's digitizer card. Unlike file-like data, stream-like data has
no well defined start and stop points. One can of course take a series
of files, each containing a chunk of a time stream, and treat them as a
stream.

Pipelines
^^^^^^^^^

A *pipeline* is a programming implementation where data processing
elements are daisy-chained together and stream data are pushed through
them -- like water through a pipe. The data processing elements all run
at once, each processing different bits of data which "flow" through the
pipeline.

Pipelining can significantly improve a code's performance (image a road
system with only one car on it at one time: that would be an
'unpipelined' transport system!). Nevertheless, if there is a
particularly slow processing element, the flow of the pipeline will be
limited by that element's output data rate-- this is known as a
*bottleneck*.

Ring buffers
^^^^^^^^^^^^

A common way to implement pipelines is to connect data processing
elements together with buffer memory storage. Bifrost uses this
approach, and the specific implementation is known as a *ring buffer*,
or just *ring*. The ring is shared between processing elements: one
element writes to the ring, while the other processing element reads
from the ring.

Blocks
^^^^^^

In Bifrost, anything that does something to a data stream is called a
*block*. Conceptually, there are three kinds of blocks: \* *tasks*: A
block that reads from a ring, transforms the data in some way, and
writes it out to an output ring. \* *source*: A block that generates
data, or loads it from outside the pipeline (e.g. a file or an Ethernet
stream), and writes it to an output ring. \* *sink*: A block that reads
data from an input ring and plots it, or writes it to file, or generally
does something without any pipeline output.

A simple pipeline would be a source block (e.g. file read), connected to
a task block (compute average), connected to a sink block (plot time
series of moving averaged data).


More
----

Further explanation of Bifrost's concepts is detailed
in the paper: https://arxiv.org/abs/1708.00720
