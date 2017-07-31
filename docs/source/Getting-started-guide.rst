.. raw:: html

Getting started guide
=====================

.. raw:: html

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

numpy, matplotlib, contextlib2, simplejson, pint, graphviz, ctypesgen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have already installed pip, this step should be as simple as
``pip install --user numpy matplotlib contextlib2 simplejson pint graphviz git+https://github.com/davidjamesca/ctypesgen.git@3d2d9803339503d2988382aa861b47a6a4872c32``.

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

Other Dependencies
^^^^^^^^^^^^^^^^^^

- exuberant-ctags
- Basic build tools (make, gcc, etc.)

On Ubuntu, the following command should grab everything you need: 
    ``sudo apt-get install build-essential software-properties-common exuberant-ctags``

Bifrost install
~~~~~~~~~~~~~~~

Now you are ready to install Bifrost. Clone the GitHub master branch
with

``git clone https://github.com/ledatelescope/bifrost``.

You will want to edit ``user.mk`` to suit your system. For example, if
you are not working with GPUs, uncomment the line:

``#NOCUDA   = 1 # Disable CUDA support``.

Now you can call ``make``, and ``make install`` to install
Bifrost.

Trying to call ``import bifrost`` inside of a Python program will tell
you if your install was successful or not.
