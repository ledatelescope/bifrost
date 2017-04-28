Python API
==========

Basic Syntax
------------

As described on the [[Home]] page, Bifrost is made up of blocks, rings,
and pipelines. Blocks embody *black box processes*, and rings connect
these blocks together. A network of blocks and rings is called a
pipeline. Bifrost's Python API mirrors these concepts very closely.

Let's start on an example: here we will perform an FFT on an array, and dump the result to a text file.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following code generates a list of three blocks: a ``TestingBlock``,
which takes a list of numbers during its initialization (which we give
as ``[1, 2, 3]``, an ``FFTBlock``, which performs a one dimensional FFT
on our input, and a ``WriteAsciiBlock``, which dumps everything given to
it into a text file (which we name as ``'logfile.txt'``).

These blocks are created into a sublist, where they are proceeded by a
subsublist of input rings, and a subsublist of output rings. The
``TestingBlock`` gets an output ring which we arbitrarily name
``'my array ring'``, the ``FFTBlock`` gets the same ring for an input
ring, and puts its results into an output ring which we name
``'fft output ring'`` and ``WriteAsciiBlock`` gets an input ring of the
same name.

This list of blocks, inputs, and outputs, is fed into a ``Pipeline``
object. Calling ``my_pipeline.main()`` then initiates the pipeline.

.. code:: python

    from bifrost.block import TestingBlock, FFTBlock, WriteAsciiBlock, Pipeline

    my_blocks = []
    my_blocks.append([TestingBlock([1, 2, 3]), [], ['my array ring']])
    my_blocks.append([FFTBlock(), ['my array ring'], ['fft output ring']])
    my_blocks.append([WriteAsciiBlock('logfile.txt'), ['fft output ring'], []])
    my_pipeline = Pipeline(my_blocks)
    my_pipeline.main() #Turn on the pipeline!

A file named ``'logfile.txt'`` should now be created and filled with the
result of our FFT.

As you can see, creating a high-throughput Bifrost pipeline from
previously written blocks is a trivial process. This is the great thing
about Bifrost: once you have modularized your functions into blocks, you
can connect them seamlessly into a pipeline, and have your data streamed
through in real-time.

Blocks
------

.. code:: python

    #Dictionaries are the form of "key": "value" pairs. If 
    # I create a dictionary such as x={"key": "value"}, 
    # then calling x["key"] produces "value".
    #I will use this key:value terminology in the 
    # following demo.

    #Always use MultiTransformBlock. I generalized it
    # to work for SourceBlock and SinkBlocks as well,
    # but have yet to rename it to be "Block."
    class DStackBlock(MultiTransformBlock):
        """Block which performs numpy's dstack operation on rings"""

        #This section of code defines the INTERNAL ring names. These
        # are internal to the block. Whenever you create an 
        # instance of this block, it will use these ring names
        # inside of it to organize itself, regardless of the pipeline.
        ring_names = {
            'in_1': "Ring containing the arrays which are stacked first",
            'in_2': "Ring containing the arrays which are stacked second",
            'out': "Outgoing ring containing the stacked array"}

        #This dictionary has keys as the ring names. Note that
        # input rings are prefixed as "in", and output
        # prefixed by "out", for the pipeline to help
        # you debug misplaced rings. 
        #The values are descriptions---these are arbitrary and
        # can be empty strings if you wish. Future pipelines
        # may use those descriptions in dot diagrams or documentation
        # generation.

        def __init__(self):
            #This is called when you first initialize the 
            # block before putting it your pipeline. Use
            # the initialization to save any user-defined
            # parameters which will affect the algorithm.

            #Call this to allow the MultiTransformBlock's
            # initialization to take place, which activates
            # some necessary functions.
            super(DStackBlock, self).__init__()

        def load_settings(self):
            #This function gets automatically called (whether
            # you choose to define it or not) by 
            # MultiTransformBlock. It is called after reading
            # in the input ring headers, and before setting
            # the gulp_sizes. Use it to calculate these sizes.
            # This function is also called before setting
            # the output ring's header. Use it to calculate
            # the output header, and assign it to self.header['ring'].
            #Note that this function will be called again 
            # each time a new sequence of the ring is loaded.

            #Here is a safety check on the defined shape and
            # datatype of the incoming data. It makes
            # sure that the algorithm will work on the rings
            # in question.
            assert self.header['in_1']['shape'] == self.header['in_2']['shape']
            assert self.header['in_1']['dtype'] == self.header['in_2']['dtype']

            #Here I (initially) set the output header to be
            # identical to the first input header. This is
            # to get the data type and blanket-copy any other
            # parameters that may be important to algorithms
            # down-the-pipe.
            self.header['out'] = dict(self.header['in_1'])

            #Here I calculate the require input gulp_sizes
            # in order to capture one 'shape' per gulp.
            self.gulp_size['in_1'] = np.product(self.header['in_1']['shape'])*self.header['in_1']['nbit']//8
            self.gulp_size['in_2'] = self.gulp_size['in_1']

            #Here I calculate the output shape. This is
            # a dstack command, so I am adding an extra
            # dimension to my data. This dimension is
            # 2 in length, as we have 2 input arrays.
            outgoing_shape = list(self.header['in_1']['shape'])
            outgoing_shape.append(2)

            #Now I put this outgoing shape into the output
            # ring, and calculate the gulp_size needed
            # to output this much data per gulp.
            self.header['out']['shape'] = outgoing_shape
            self.gulp_size['out'] = self.gulp_size['in_1']*2
        def main(self):
            #This function is called once by the pipeline
            # once all of the rings are in place. It actually
            # gets called BEFORE load_settings, but 
            # the first time you open your input rings, that
            # function gets called. 
            #Define your algorithm with this function.

            #Here I do a blanket-read/write for loop. This
            # is necessary if you want to do your reading
            # and writing in synchrony. The self.read command
            # takes the names of the input rings, and 
            # generates input_spans each loop. The self.write
            # command does the same for output rings. The 
            # self.izip command turns this statement into 
            # a single generator, so you only need one for loop. 
            #Make sure to order your input spans and output spans
            # according to how they are placed in the read and
            # write calls.
            for inspan1, inspan2, outspan in self.izip(
                    self.read('in_1', 'in_2'),
                    self.write('out')):
                #Inside this loop, I have my input data
                # in the form of inspan1, and inspan2. 
                # The output data allocation is in outspan.
                #All of the datatypes are loaded automatically
                # based on the header. All I do now is process the
                # input spans, and copy them into the output span.
                outspan[:] = np.dstack((
                        inspan1.reshape(self.header['in_1']['shape']),
                        inspan2.reshape(self.header['in_2']['shape']))).ravel()[:]
                #The use of the indices [:] here is very important. It
                # is a copy command. Stating outspan=... would 
                # reassign the name "outspan" instead of copying
                # data, so the output ring would receive nothing.

Pipelines
---------

.. code:: python

    #Dummy function to generate data
    def generate_ten_arrays():
        for i in range(10):
            yield np.array([1, 2, 3])

    #Function to print numpy arrays
    def pprint(array):
        print array

    #The following list will contain all 
    # of the initialized blocks, along 
    # with their 'connections', which 
    # define how blocks connect to eachother
    blocks = []

    #Each block is defined as a two-element
    # sublist (a list within the list of blocks). 
    # The first part of the list is
    # an object---like the one I created above.
    # When you create the object, you call the
    # __init__ function, so define any algorithm
    # parameters in that part.
    blocks.append([
        NumpySourceBlock(generate_ten_arrays()),
        {'out_1':0}])
    #The second part of this sublist is a dictionary
    # of connections. 'out_1'---the key, refers
    # to the internal ring definition in
    # the block. This would be one of the ring_name's
    # that I defined above for DStackBlock.
    # It gets used internally for organizing 
    # the headers, gulp_sizes, and so on.
    #The 0 is the value. It is the EXTERNAL (!)
    # ring name. It is what we want to call the 
    # ring in the context of this pipeline. 
    # If you are confused because it is a number,
    # note that you could also use a string in
    # place of the 0. For example 'raw_input' 
    # is a valid name. You could define
    # your own Ring() object, and use that in place.
    # So long as you are consistent. Each ring
    # should only be named once as an output, and 
    # as many times as you want as an input. Bifrost
    # will let you know if you have used a ring
    # twice as an output. 


    #Here I generate an identical block to feed 
    # another ring. Note that the INTERNAL name
    # of the ring is the same as above, as it 
    # is only a marker for the block,
    # but the EXTERNAL name of the ring, the name
    # that our pipeline will use, is different. 
    blocks.append([
        NumpySourceBlock(generate_ten_arrays()),
        {'out_1':1}])
    #Here I defined the external ring to be 1. 
    # We must refer to that name later on.


    #Here we use the block we defined above
    # to stack these inputs together. Note that
    # the 'in_1' and 'in_2' keys are the same
    # names we used in the block definition
    # above. We have assigned these inputs to
    # the rings 0 and 1 in our pipeline, which 
    # are being outputted by the above generators.
    blocks.append([
        DStackBlock(),
        {'in_1': 0, 'in_2': 1, 'out': 2}])
    #For the output, we define the external
    # ring to be 2. This will be used later.

    #Finally, I want to see if the pipeline has
    # worked, so I print all the outgoing arrays.
    # NumpyBlock by default has inputs=1, outputs=1,
    # so for this block to be satisfied with
    # no output data, we set outputs=0 in the call to
    # the block. 
    blocks.append([
        NumpyBlock(pprint, outputs=0),
        {'in_1': 2}])
    #NumpyBlock and NumpySourceBlock have
    # their INTERNAL rings automatically
    # defined as 
    # in_1, in_2, ... and out_1, out_2...,
    # for as many as you wish to make. 
    #Here we set the only input to be 
    # 2 --- the ring that is outputted
    # by the DStackBlock.

    #Create a pipeline with these blocks, and
    # execute it (main). 
    Pipeline(blocks).main()

