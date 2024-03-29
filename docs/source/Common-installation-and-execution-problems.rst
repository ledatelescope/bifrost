Common Installation and Execution Problems
==========================================

Do you have a problem that is not documented below? No matter how
trivial the problem might be, please raise an issue.

(The program hangs)
-------------------

This is probably because a Bifrost pipeline was started with an infinite
timeout, and some blocks are not ending themselves. Quit the program (by
ctrl-\\), and make sure every block in your pipeline is reading/writing
to its rings as it should.

OSError: Can't find library with name libbifrost.so
---------------------------------------------------

This means that the bifrost Python wrapper can't find your Bifrost
installation. Whatever library folders it searches for Bifrost, you
do not have the libbifrost.so file there. To fix this, type

``echo $LD_LIBRARY_PATH``

at your command line. If none of these folders contain the Bifrost
installation (which you may have specified with configure), you have found
the problem. Perform

``export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/my/bifrost/installation``,

where ``/my/bifrost/installation`` is the folder where you installed the
Bifrost "lib" (this location will also be reported as part of `make install`).
This should add Bifrost to the wrapper's search path.

OSError: libcudart.so.x.0: cannot open shared object file: No such file or directory
------------------------------------------------------------------------------------

Similar to the above error. You need to add the CUDA libraries to the
LD\_LIBRARY\_PATH search path.

For example,
``export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/my/cuda/installation/lib64``
