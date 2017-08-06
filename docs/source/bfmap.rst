Fast GPU math using bfMap
=========================

A key under-the-hood part of ``bifrost`` is the ``map`` function, that
provides a simple way to do fast arithmetic operations on the GPU. To
get acquainted, we will ignore all of the pipeline infrastructure that
bifrost provides, and just call the map function directly.

Let's suppose you have two ``ndarrays`` and wish to add them together on
the GPU. Here is how you would do that with ``map``:

.. code:: python

    import bifrost as bf

    # Create two arrays on the GPU, A and B, and an empty output C
    a = bf.ndarray([1,2,3,4,5], space='cuda')
    b = bf.ndarray([1,0,1,0,1], space='cuda')
    c = bf.ndarray(np.zeros(5), space='cuda')

    # Add them together
    bf.map("c = a + b", data={'c': c, 'a': a, 'b': b})
    print c
    # ndarray([ 2.,  2.,  4.,  4.,  6.])

The map function figures out what to do based on the string you give it.
These would also work:

.. code:: python

    bf.map("c = a - b", data={'c': c, 'a': a, 'b': b})
    bf.map("c = a * b", data={'c': c, 'a': a, 'b': b})
    bf.map("c = a / b", data={'c': c, 'a': a, 'b': b})
    bf.map("c += a + b", data={'c': c, 'a': a, 'b': b})
    bf.map("c /= a + b", data={'c': c, 'a': a, 'b': b})
    bf.map("c *= a + b", data={'c': c, 'a': a, 'b': b})

Getting deeper requires a look at the docstring:

.. code:: python

    def map(func_string, *args, **kwargs):
        """Apply a function to a set of ndarrays.

        Args:
          func_string (str): The function to apply to the arrays, as a string (see
                       below for examples).
          data (dict): Map of string names to ndarrays or scalars.
          axis_names (list): List of string names by which each axis is referenced
                       in func_string.
          shape:       The shape of the computation. If None, the broadcast shape
                       of all data arrays is used.
          func_name (str): Name of the function, for debugging purposes.
          extra_code (str): Additional code to be included at global scope.
          block_shape: The 2D shape of the thread block (y,x) with which the kernel
                       is launched.
                       This is a performance tuning parameter.
                       If NULL, a heuristic is used to select the block shape.
                       Changes to this parameter do _not_ require re-compilation of
                       the kernel.
          block_axes:  List of axis indices (or names) specifying the 2 computation
                       axes to which the thread block (y,x) is mapped.
                       This is a performance tuning parameter.
                       If NULL, a heuristic is used to select the block axes.
                       Values may be negative for reverse indexing.
                       Changes to this parameter _do_ require re-compilation of the
                       kernel.

        Note:
            Only GPU computation is currently supported.

        Examples::

          # Add two arrays together
          bf.map("c = a + b", {'c': c, 'a': a, 'b': b})

          # Compute outer product of two arrays
          bf.map("c(i,j) = a(i) * b(j)",
                 {'c': c, 'a': a, 'b': b},
                 axis_names=('i','j'))

          # Split the components of a complex array
          bf.map("a = c.real; b = c.imag", {'c': c, 'a': a, 'b': b})

          # Raise an array to a scalar power
          bf.map("c = pow(a, p)", {'c': c, 'a': a, 'p': 2.0})

          # Slice an array with a scalar index
          bf.map("c(i) = a(i,k)", {'c': c, 'a': a, 'k': 7}, ['i'], shape=c.shape)
        """

Let's look a bit closer at that outer product example. Here, by
convention of summation notation, the indexes 'i', 'j' on the two arrays
A and B, create an outer product. A full example:

.. code:: python

    import bifrost as bf

    # Create two arrays on the GPU, A and B, and an empty output C
    a = bf.ndarray([1,2,3,4,5], space='cuda')
    b = bf.ndarray([1,0,1,0,1], space='cuda')
    c = bf.ndarray(np.zeros((5, 5)), space='cuda')

    # Compute outer product
    bf.map("c(i,j) = a(i) * b(j)",
           axis_names=['i', 'j'],
           data={'c': c, 'a': a, 'b': b})
    print c

    # ndarray([[ 1.,  0.,  1.,  0.,  1.],
    #          [ 2.,  0.,  2.,  0.,  2.],
    #          [ 3.,  0.,  3.,  0.,  3.],
    #          [ 4.,  0.,  4.,  0.,  4.],
    #          [ 5.,  0.,  5.,  0.,  5.]])

The first example of ``c = a + b`` could also be written more explicitly as:

.. code:: python

    bf.map("c(i) = a(i) + b(i)", axis_names=['i'], data={'c': c, 'a': a, 'b': b})

Note, however, that implicit indexing should be preferred where possible, as
explicit indexing may exhibit worse performance.
