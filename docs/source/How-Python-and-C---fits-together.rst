How bifrost fits together
=========================

*(aka seeing C in Python pipelines)*

Bifrost is based on two maxims: 1) by itself, C sucks. 2) by itself,
Python sucks.

Bifrost bridges the two languages using some clever python libraries, a
customized numpy array class, and enforcing some pipeline friendly
design choices.

A trip to the C
---------------

Suppose you have some custom C code which you want to integrate into a
bifrost pipeline. Here's a nice and simple snippet that adds two arrays
together:

.. code:: C

        for(int i=0; i < nelements; i+=1)
        {
            x[i] = x[i] + y[i];
        }

To get this into bifrost and use it in Python, you need to 'bifrostify'
this code to accept bifrost's special ``ndarray`` class.

Bifrost Python ndarrays
~~~~~~~~~~~~~~~~~~~~~~~

Bifrost has a special array class in python that plays nicely with C:

.. code:: python

        import bifrost as bf
        a = bf.ndarray([1,2,3,4,5,6,7,8,9,10],  dtype='f32')
        b = bf.ndarray([2,3,4,5,6,7,8,9,10,11], dtype='f32')

This ``ndarray`` object is very similar to the ``numpy.array``, but it
has a special method, ``.as_BFarray()``. Essentially, ``as_BFarray``
returns a pointer to the numpy array's memory address, along with some
other useful stuff.

.. code:: python

        z = a.as_BFarray()
        
        # Tab complete will show you this object has:
        z.big_endian z.dtype      z.shape
        z.conjugated z.immutable  z.space
        z.data       z.ndim       z.strides

Bifrost C++ BFarray
~~~~~~~~~~~~~~~~~~~

In the bifrost C++ code, there is a matching ``BFarray`` that makes
interfacing with Python straightforward. This is essentially a struct
that provides the same info as the Python ``ndarray``.

A usage example is worth a thousand words, so here is our simple snippet
from before, after it's been bifrostified:

.. code:: C

    BFstatus AddStuff(BFarray *xdata, BFarray *ydata)
    {
        long nelements = num_contiguous_elements(xdata);

        float* x = (float *)xdata->data;
        float* y = (float *)ydata->data;

        for(int i=0; i < nelements; i +=1)
        {
           x[i] = x[i] + y[i];
        }

        return BF_STATUS_SUCCESS;
    }

A full code example (with headers etc) can be found at the end.

Using your C++ code in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now comes the cool part. You'll need to edit three files:

1) create ``add_stuff.cpp`` in ``src``
2) create ``add_stuff.h`` in ``src/bifrost``
3) edit ``src/Makefile`` and add ``add_stuff.o`` in the LIBBIFROST\_OBJS
   part

Now, run ``make`` and ``make install`` from the root directory to
rebuild the ``libbifrost.so`` library.

Once this is recompiled, open up iPython and try this:

.. code:: python

    import bifrost as bf
    from bifrost.libbifrost import _bf

    b = bf.ndarray([1,2,3,4,5,6,7,8,9,10], dtype='f32')
    a = bf.ndarray([1,2,3,4,5,6,7,8,9,10], dtype='f32')
    _bf.AddStuff(a.as_BFarray(), b.as_BFarray())
    print a

That is: your ``AddStuff`` function is available in python via some
`ctypesgen <https://github.com/davidjamesca/ctypesgen>`__ magic. You
can't just pass ``a`` and ``b`` by themselves, but you can send their
``as_BFarray()`` output.

Wrapping up
~~~~~~~~~~~

Bravo, you've managed to run C++ code in bifrost! All the rest of the
pipeline stuff is just python (to be continued...)

Example C++ codes
-----------------

add\_stuff.cpp
~~~~~~~~~~~~~~

.. code:: C

    #include <bifrost/cpu_add.h>
    #include <bifrost/array.h>
    #include <bifrost/common.h>
    #include <bifrost/ring.h>
    #include <utils.hpp>
    #include <stdlib.h>
    #include <stdio.h>
    #include <iostream>

    extern "C" {
    BFstatus AddStuff(BFarray *xdata, BFarray *ydata)
    {
        long nelements = num_contiguous_elements(xdata);

        float* x = (float *)xdata->data;
        float* y = (float *)ydata->data;

        for(int i=0; i < nelements; i +=1)
        {
           x[i] = x[i] + y[i];
        }

        return BF_STATUS_SUCCESS;
    }

    }

bifrost/add\_stuff.h
~~~~~~~~~~~~~~~~~~~~

.. code:: C

    #include <bifrost/common.h>
    #include <bifrost/array.h>

    extern "C" {

    BFstatus AddStuff(BFarray *xdata, BFarray *ydata);

    }

