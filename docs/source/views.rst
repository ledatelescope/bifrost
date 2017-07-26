Views
=====

Views in bifrost are conceptually similar to the ``numpy.view()``, in
that it changes the interpretation of the data in a ring, without
copying or changing the underlying data. These are useful for preparing
your data for later processing. The views provided are:

-  ``views.split_axis``: reshape an array to split an axis into two:
   ``[freq, time]`` becomes ``[freq, time, window]`` to chunk up the
   data in preparation for an FFT.
-  ``views.merge_axis``: flatten two axes of an array:
   ``[channel, freq]`` becomes ``[freq]``.
-  ``views.rename_axis``: change the label on an axis:
   ``[one_over_wavelength]`` becomes ``[freq]``
-  ``views.expand_dims``: add an extra dimension:
   shape\ ``[-1, 10, 100]`` becomes ``[-1, 10, 1, 100]``
-  ``views.custom``: apply a custom view

As an example, this reads two "time steps" from a GUPPI RAW file, and
squishes it into one longer time step:

.. code:: python

    # Read from guppi raw file
    b_guppi   = blocks.read_guppi_raw(filelist, core=1, buffer_nframe=4)
    b_gup2    = views.rename_axis(b_guppi, 'freq', 'channel')
    
    # Buffer up two time steps & reshape to allow longer FFT
    b_gup2    = views.split_axis(b_gup2, axis='time', n=n_chunks, label='time_chunk')
    b_gup2    = blocks.transpose(b_gup2, axes=['time', 'channel', 'time_chunk', 'fine_time', 'pol'], buffer_nframe=1)
    b_gup2    = views.merge_axes(b_gup2, 'time_chunk', 'fine_time', label='fine_time')
