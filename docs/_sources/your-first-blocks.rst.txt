Your first blocks
=================

If you're lucky, all the bits and pieces you need to build your pipeline
are already blocks in bifrost. More likely, you have a particular
algorithm you'd like to plug in. For this, you'll need to create a new
block. This basically comes down to definining a class with two
mandatory functions: ``on_sequence()``, which is called whenever a new
sequence is started, and ``on_data()``, which is called whenever there
is new data ready at the ring buffer.

TransformBlock
--------------

A ``TransformBlock`` reads data from one ring buffer, does something to
it, and then writes it out. Take this example, which adds a
runtime-specified value to every element in the ring buffer:

.. code:: python

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
            
            idata = ispan.data 
            odata = ospan.data
            
            odata[...] = idata + self.n_to_add
            return out_nframe

This is a new class that subclasses ``TransformBlock``. First, let's
look at the ``__init__`` method. This takes two parameters:

-  ``iring`` - an input ring buffer. This argument is required, but
   bifrost handles the setup of the ring buffers.
-  ``n_to_add`` - this is a new argument that we've added ourselves.

The ``super(UselessAddBlock, self)`` call passes the ``iring``, and
optional ``*args`` and ``**kwargs`` on to the parent class for
initialization.

Next, we have an ``on_sequence()`` method that is called whenever a new
sequence arrives. For example, reading a new file may trigger a new
sequence, with new metadata in the header. The ``on_sequence()`` method
requires an ``iseq`` argument, and needs to output its own sequence
header. The ``deepcopy`` is (currently) required to make sure the
original dictionary isn't passed on by accident. Note that all we are
doing here is changing the name by appending a string.

Finally, there's the ``on_data()`` method that requires an ``ispan`` and
``ospan`` argument, for reading and writing data in and out of the ring
buffers. ``on_data()`` needs to return the number of frames in the
output span.

SinkBlock
---------

A ``SinkBlock`` also needs an ``on_sequence()`` and ``on_data()``
method, but doesn't need to output anything, so neither method should
return anything. Here is a simple block to print stuff to screen:

.. code:: python

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

Note that ``on_data()`` shall not have an ``ospan`` argument!

SourceBlock
-----------

The ``SourceBlock`` is a little trickier to get up and going as it
requires some fun with `context
managers <https://jeffknupp.com/blog/2016/03/07/python-with-context-managers/>`__.
The source block also has the important task of setting up all the
metadata required to make ``bifrost`` work -- a little extra effort at
the start allows useful metadata to propagate through the full pipeline,
simplifying future blocks.

Here is the source code for the
`binary\_io <https://github.com/ledatelescope/bifrost/blob/master/python/bifrost/blocks/binary_io.py>`__
block to read from data saved using the useful ``numpy.tofile()``:

.. code:: python

    class BinaryFileRead(object): 
        """ Simple file-like reading object for pipeline testing
        
        Args:
            filename (str): Name of file to open
            dtype (np.dtype or str): datatype of data, e.g. float32. This should be a *numpy* dtype,
                                     not a bifrost.ndarray dtype (eg. float32, not f32)
            gulp_size (int): How much data to read per gulp, (i.e. sub-array size)
        """
        def __init__(self, filename, gulp_size, dtype):
            super(BinaryFileRead, self).__init__()
            self.file_obj = open(filename, 'r')
            self.dtype = dtype
            self.gulp_size = gulp_size
            
        def read(self):
            d = np.fromfile(self.file_obj, dtype=self.dtype, count=self.gulp_size)
            return d
            
        def __enter__(self):
            return self
        
        def close(self):
            pass
        
        def __exit__(self, type, value, tb):
            self.close()


    class BinaryFileReadBlock(bfp.SourceBlock):
        """ Block for reading binary data from file and streaming it into a bifrost pipeline
        
        Args:
            filenames (list): A list of filenames to open
            gulp_size (int): Number of elements in a gulp (i.e. sub-array size)
            gulp_nframe (int): Number of frames in a gulp. (Ask Ben / Miles for good explanation)
            dtype (bifrost dtype string): dtype, e.g. f32, cf32
        """
        def __init__(self, filenames, gulp_size, gulp_nframe, dtype, *args, **kwargs):
            super(BinaryFileReadBlock, self).__init__(filenames, gulp_nframe, *args, **kwargs)
            self.dtype = dtype
            self.gulp_size = gulp_size
            
        def create_reader(self, filename):
            print "Loading %s" % filename
            # Do a lookup on bifrost datatype to numpy datatype
            dcode = self.dtype.rstrip('0123456789')
            nbits = int(self.dtype[len(dcode):])
            np_dtype = name_nbit2numpy(dcode, nbits)
            
            return BinaryFileRead(filename, self.gulp_size, np_dtype)
             
        def on_sequence(self, ireader, filename):        
            ohdr = {'name': filename,
                    '_tensor': {
                            'dtype':  self.dtype,
                            'shape':  [-1, self.gulp_size],
                            }, 
                    }
            return [ohdr]
        
        def on_data(self, reader, ospans):
            indata = reader.read()
            
            if indata.shape[0] == self.gulp_size:
                ospans[0].data[0] = indata
                return [1]
            else:
                return [0]

As ``bifrost`` requires a reader with baked-in context management, we
have explicitly created a ``BinaryFileRead`` object that has an
``__enter__`` and ``__exit__`` method; these are *mandatory*. This also
has a crucially important ``read()`` function, to read data into the
ring.

The second class, ``BinaryFileReadBlock`` is doing the reading, and
again has an ``on_sequence()`` and ``on_data()`` method. There is also a
mandatory ``create_reader`` method, that does some setup, in this case
of the file handler.

The \_tensor dict
~~~~~~~~~~~~~~~~~

The ``on_sequence()`` method has an important job to setup the header
metadata. This requires a mandatory (and *unique* ``name``) and making a
``_tensor`` dictionary that describes the dimensions and datatype of the
data in each span:

.. code:: python

    ohdr = {'name': filename,
             '_tensor': {
                         'dtype':  self.dtype,
                         'shape':  [-1, self.gulp_size],
                         }, 
             }

A complete pipeline
-------------------

Putting it all together, we have this complete pipeline below, which
reads from a file, adds something to it with out ``UselessAddBlock``,
and then prints out some diagnostic info with our ``PrintStuffBlock``.
This is also available in the
`testbench <https://github.com/telegraphic/bifrost/tree/master/testbench>`__
directory in the repository.

.. code:: python

    """
    # your_first_block.py

    This testbench initializes a simple bifrost pipeline that reads from a binary file,
    and then writes the data to an output file. 
    """
    import os
    import numpy as np
    import bifrost.pipeline as bfp
    from bifrost.blocks import BinaryFileReadBlock
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
            
            idata = ispan.data 
            odata = ospan.data
            
            odata[...] = idata + self.n_to_add
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
