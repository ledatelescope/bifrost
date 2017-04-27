Writing to output ring
----------------------

A block with an output ``Ring`` writes data to the output in the
following way:

.. code:: python

    with self.oring.begin_writing() as oring:
        with oring.begin_sequence(filename, header=ohdr) as osequence:
            with osequence.reserve(gulp_nbyte) as wspan:
                data = data.view('uint8').ravel()
                    wspan.data[0][:] = data

The ``with`` statements ensure that the ``__exit__`` routine runs to
close files / locations. [ MORE DETAIL LATER]
