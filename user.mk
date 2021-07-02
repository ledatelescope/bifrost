
CXX           ?= g++
NVCC          ?= nvcc
LINKER        ?= g++
CXX_FLAGS     ?= -O3 -Wall 
NVCC_FLAGS    ?= -O3 -Xcompiler "-Wall" --ptxas-options=-v
LINKER_FLAGS  ?=


GPU_SHAREDMEM ?= 16384 # GPU shared memory size

#GPU_ARCHS     ?= 52
GPU_ARCHS     ?= 61

CUDA_HOME     ?= /usr/local/cuda
CUDA_LIBDIR   ?= $(CUDA_HOME)/lib
CUDA_LIBDIR64 ?= $(CUDA_HOME)/lib64
CUDA_INCDIR   ?= $(CUDA_HOME)/include

ALIGNMENT ?= 4096 # Memory allocation alignment

PSRDADA_PATH  ?= /home/npsr/software/psrdada/latest
BUILDDP4A   = 1  # Build beanfarmer + xcorr_lite
BUILDROMEIN   = 1  # Build Romein kernels
#NODEBUG    = 1 # Disable debugging mode (use this for production releases)
#NOUDPSOCKET = 1  # Disable UDP code compilation
TRACE      = 1 # Enable tracing mode (generates annotations for use with nvprof/nvvp)
#NOCUDA     = 1 # Disable CUDA support
ANY_ARCH   = 1 # Disable native architecture compilation
#CUDA_DEBUG = 1 # Enable CUDA debugging (nvcc -G)
NUMA       = 1 # Enable use of numa library for setting affinity of ring memory
HWLOC      = 1 # Enable use of hwloc library for memory binding in udp_capture
#VMA        = 1 # Enable use of Mellanox libvma in udp_capture
