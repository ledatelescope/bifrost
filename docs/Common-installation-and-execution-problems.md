Do you have a problem that is not documented below? No matter how trivial the problem might be, please raise an issue.

#### ImportError: No module named pyclibrary

You have not installed PyCLibrary, or you are using two different python installations (i.e., one installed via apt-get, and one installed from source in a local directory, or one in a virtual environment). Make sure you are using the same Python to install libraries as you are to run programs. Get PyCLibrary from [here](https://github.com/MatthieuDartiailh/pyclibrary).

#### OSError: ..../lib/libbifrost.so: undefined symbol: cudaFreeHost

At the make step, nvcc did not link cudaFreeHost into libbifrost.so. You should make sure that config.mk and user.mk are set up for your system, and that your nvcc compiler can compile other CUDA programs. If you are still having trouble, raise an issue.

#### OSError: Can't find library with name libbifrost.so

This means that PyCLibrary can't find your Bifrost installation. Whatever library folders it searches for Bifrost, you do not have the libbifrost.so file there. To fix this, type 

`echo $LD_LIBRARY_PATH`

 at your command line. If none of these folders contain the Bifrost installation (which you specified in config.mk), you have found the problem. Perform 

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/my/bifrost/installation`,

where `/my/bifrost/installation` is the folder where you installed the Bifrost "lib" (in config.mk, this folder is given as `INSTALL_LIB_DIR`). This should add Bifrost to the PyCLibrary search path.