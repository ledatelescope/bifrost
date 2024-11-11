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

You will need a relatively new gcc and CUDA - we have used Bifrost with gcc 4.8 and CUDA 8,
but higher ones should also work.

Python dependencies
~~~~~~~~~~~~~~~~~~~

*Bifrost is compatible with Python >3.6.*

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
``pip install --user numpy matplotlib contextlib2 simplejson pint graphviz ctypesgen==1.0.2``.

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
the operating system-specific instructions to install. Be sure to install a
kernel driver that works with the version of toolkit that you are using since 
version mismatches can lead to runtime errors.

The table below indicates which CUDA toolkit and kernel driver versions Bifrost
has been tested against.

.. csv-table::
   :header: "OS","Linux Kernel","Driver Version","GPU","Toolkit","Status"
   :widths: 15,18,12,18,12,35

   "Ubuntu 20.04","5.4.0-177-generic","520.61.05","RTX 2080Ti","11.0.3","Working"
   "Ubuntu 20.04","5.4.0-177-generic","520.61.05","RTX 2080Ti","11.1.1","Working"
   "Ubuntu 20.04","5.4.0-177-generic","520.61.05","RTX 2080Ti","11.2.2","Working"                       
   "Ubuntu 20.04","5.4.0-177-generic","520.61.05","RTX 2080Ti","11.3.1","Working"
   "Ubuntu 20.04","5.4.0-186-generic","470.239.06","Quadro K2200","11.4.4","Working"
   "Ubuntu 20.04","5.4.0-186-generic","495.29.05","Quadro K2200","11.5.2","Working"
   "Ubuntu 18.04","4.15.0-88-generic","510.85.02","A4000","11.6.124","Working"
   "Ubuntu 18.04","4.15.0-88-generic","510.85.02","RTX 2080Ti","11.6.124","Working"
   "Ubuntu 20.04","4.4.0-174-generic","525.147.05","Titan RTX","11.6.55","Working"
   "Ubuntu 20.04","5.4.0-186-generic","520.61.05","Quadro K2200","11.8.0","Known FIR and FFT Problems"
   "Debian 12","6.1.0-21-amd64","525.147.05 ","Quadro K2200","12.0.0","Working"
   "Ubuntu 20.04","5.4.0-147-generic","525.125.06","A5000","12.0.140","Working"
   "Ubuntu 20.04","5.4.0-144-generic","525.125.06","A5000","12.0.140","Working"
   "Ubuntu 20.04","5.4.0-144-generic","525.147.05","A4000","12.0.140","Working"
   "Debian 12","6.1.0-21-amd64","530.30.02","Quadro K2200","12.1.1","Working"
   "Debian 12","6.1.0-21-amd64","535.104.05","Quadro K2200","12.2.2","FFT bug: zeroed out data"
   "Debian 12","6.1.0-21-amd64","545.23.08","Quadro K2200","12.3.2","Working"
   "Debian 12","6.1.0-21-amd64","550.54.15","Quadro K2200","12.4.1","Working"
   "Ubuntu 22.04","5.15.0-106-generic","555.42.06","GTX 980","12.5","Working"

Other Dependencies
^^^^^^^^^^^^^^^^^^

- universal-ctags
- Basic build tools (make, gcc, etc.)

On Ubuntu, the following command should grab everything you need: 
    ``sudo apt-get install build-essential software-properties-common universal-ctags``

Bifrost install
~~~~~~~~~~~~~~~

Now you are ready to install Bifrost. Clone the GitHub master branch
with

``git clone https://github.com/ledatelescope/bifrost``.

You will want to run `configure` to tailor Bifrost to you system.  At the end of
`configure` you will get a summary of how Bifrost will be built:

```
...
config.status: creating src/bifrost/config.h
config.status: executing libtool commands

configure: cuda: yes - 50 52
configure: numa: yes
configure: hwloc: yes
configure: libvma: no
configure: python bindings: yes
configure: memory alignment: 4096
configure: logging directory: /dev/shm/bifrost
configure: options: native

Bifrost is now ready to be compiled.  Please run 'make'
```

Now you can call ``make``, and ``make install`` to install
Bifrost.

Trying to call ``import bifrost`` inside of a Python program will tell
you if your install was successful or not.
