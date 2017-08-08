CXX           ?= g++
NVCC          ?= nvcc
LINKER        ?= g++
CPPFLAGS      ?=
CXXFLAGS      ?= -O3 -Wall -pedantic
NVCCFLAGS     ?= -O3 -Xcompiler "-Wall" #-Xptxas -v
LDFLAGS       ?=
DOXYGEN       ?= doxygen
PYBUILDFLAGS   ?=
PYINSTALLFLAGS ?=

#GPU_ARCHS     ?= 30 32 35 37 50 52 53 # Nap time!
#GPU_ARCHS     ?= 35 52
GPU_ARCHS     ?= 35 61

CUDA_HOME     ?= /usr/local/cuda
CUDA_LIBDIR   ?= $(CUDA_HOME)/lib
CUDA_LIBDIR64 ?= $(CUDA_HOME)/lib64
CUDA_INCDIR   ?= $(CUDA_HOME)/include

ALIGNMENT ?= 4096 # Memory allocation alignment

#NODEBUG    = 1 # Disable debugging mode (use this for production releases)
#TRACE      = 1 # Enable tracing mode (generates annotations for use with nvprof/nvvp)
#NOCUDA     = 1 # Disable CUDA support
#ANY_ARCH   = 1 # Disable native architecture compilation
#CUDA_DEBUG = 1 # Enable CUDA debugging (nvcc -G)
#NUMA       = 1 # Enable use of numa library for setting affinity of ring memory
#HWLOC      = 1 # Enable use of hwloc library for memory binding in udp_capture
#VMA        = 1 # Enable use of Mellanox libvma in udp_capture
