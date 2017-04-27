.. raw:: html

   <!---
   Currently using this book of tips : https://jacobian.org/writing/great-documentation/
   -->

   <!---
   Should get a new user up and running in about 30 minutes.
   -->

Getting started guide
=====================

.. raw:: html

   <!--- For a quick demo of Bifrost to whet your appetite, please check out our online demo at [GitHub Pages](bifrost.github.io). -->
    

If you have already installed Bifrost, look below to `Creating your
first pipeline <#tutorial>`__

Installation
------------

Bifrost requires several dependencies, depending on how you want to use
it. If you don't know what you are doing, assume that you want all the
dependencies - we will walk you through this process.

Python dependencies
~~~~~~~~~~~~~~~~~~~

*Bifrost is written in Python 2.7. If you would like us to support
Python 3.x, please let us know your interest.*

`pip <https://pip.pypa.io/en/stable/>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pip is a package manager for other Python dependencies. Once you have
pip, installing additional python dependencies should be
straightforward. pip comes with setuptools, which is required for
installing Bifrost. The detailed instructions for pip can be found
`here <https://pip.pypa.io/en/stable/installing/>`__, but the basics are
as follows:

1. Download ```get-pip.py`` <https://bootstrap.pypa.io/get-pip.py>`__
2. Navigate to the download directory, and run
   ``python get-pip.py --user``, which will install a local copy of pip.
3. Check pip is working with ``pip list``, which will give the versions
   of pip and setuptools.

`NumPy <http://docs.scipy.org/doc/numpy/index.html>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have already installed pip, this step should be as simple as
``pip install numpy``. NumPy is used in the Python front end of Bifrost,
and is very helpful for quick testing of pipeline ideas.

`PyCLibrary (modified) <https://github.com/MatthieuDartiailh/pyclibrary>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyCLibrary is a Python library that will parse C files and ease the
interaction with ``ctypes``. The entire Bifrost front end is built with
it. **Do not install the pip version, as it is out of date**. You need
to get the source from
`GitHub <https://github.com/MatthieuDartiailh/pyclibrary>`__, and
install it manually:

1. Download PyCLibrary by running
   ``git clone https://github.com/MatthieuDartiailh/pyclibrary``
2. Enter the PyCLibrary folder that was just created, and run
   ``python setup.py install --user``.
3. Check that PyCLibrary installed correctly by running ``python``, and
   then trying ``import pyclibrary``. If this works, you are ready to
   go.

C++ dependencies
~~~~~~~~~~~~~~~~

`CUDA <https://developer.nvidia.com/cuda-zone>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA allows you to program your GPU from C and C++. You will need an
NVIDIA GPU to do this. If this is your first time trying out Bifrost,
and you don't have CUDA yet, we recommend that you skip this step, and
try out a CPU-only version of Bifrost. Then, once you have that first
experience, you can come back here for a speedup.

If you are ready to work with a GPU, you will want to get the newest
`CUDA toolkit <https://developer.nvidia.com/cuda-downloads>`__. Follow
the operating system-specific instructions to install.

`OpenMP <http://openmp.org/wp/openmp-compilers/>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bifrost install
~~~~~~~~~~~~~~~

Now you are ready to install Bifrost. Clone the GitHub master branch
with

``git clone https://github.com/ledatelescope/bifrost``.

You will want to edit ``user.mk`` to suit your system. For example, if
you are not working with GPUs, uncomment the line:

``#NOCUDA   = 1 # Disable CUDA support``.

Now you can call ``make``, and ``make install`` to install the C backend
of Bifrost to your computer.

After this, navigate to ``bifrost/python``, and run
``python setup.py install --user``, to install the Python interface.

Trying to call ``import bifrost`` inside of a Python program will tell
you if your install was successful or not.

Create your first pipeline
--------------------------

