CXX           ?= /opt/gcc-5.4.0/bin/g++ 
NVCC          ?= nvcc
LINKER        ?= /opt/gcc-5.4.0/bin/g++
CPPFLAGS      ?=
CXXFLAGS      ?= -O3 -Wall -pedantic -mno-avx2
NVCCFLAGS     ?= -O3 -Xcompiler "-Wall" #-Xptxas -v
LDFLAGS       ?=
DOXYGEN       ?= doxygen
PYBUILDFLAGS   ?=
PYINSTALLFLAGS ?=

GPU_ARCHS     ?= 52 61 75

GPU_SHAREDMEM ?= 16384 # GPU shared memory size

CUDA_HOME     ?= /opt/cuda-10.2.89
CUDA_LIBDIR   ?= $(CUDA_HOME)/lib
CUDA_LIBDIR64 ?= $(CUDA_HOME)/lib64
CUDA_INCDIR   ?= $(CUDA_HOME)/include

ALIGNMENT ?= 4096 # Memory allocation alignment

PSRDADA_PATH  ?= /home/npsr/software/psrdada/latest
NOUDPSOCKET = 1  # Disable UDP code compilation
BUILDDP4A   = 1  # Build beanfarmer + xcorr_lite
#NODEBUG    = 1 # Disable debugging mode (use this for production releases)
TRACE      = 1 # Enable tracing mode (generates annotations for use with nvprof/nvvp)
#NOCUDA     = 1 # Disable CUDA support
ANY_ARCH   = 1 # Disable native architecture compilation
#CUDA_DEBUG = 1 # Enable CUDA debugging (nvcc -G)
#NUMA       = 1 # Enable use of numa library for setting affinity of ring memory
#HWLOC      = 1 # Enable use of hwloc library for memory binding in udp_capture
#VMA        = 1 # Enable use of Mellanox libvma in udp_capture
