
CXX           ?= g++
NVCC          ?= nvcc
LINKER        ?= g++
CXX_FLAGS     ?= -O3 -Wall
NVCC_FLAGS    ?= -O3 -Xcompiler "-Wall" -Xptxas -v
LINKER_FLAGS  ?=

#GPU_ARCHS     ?= 30 32 35 37 50 52 53 # Nap time!
#GPU_ARCHS     ?= 35 52
#GPU_ARCHS     ?= 52
#GPU_ARCHS     ?=
GPU_ARCHS     ?= 20

CUDA_HOME     ?= /usr/local/cuda
CUDA_LIBDIR   ?= $(CUDA_HOME)/lib
CUDA_LIBDIR64 ?= $(CUDA_HOME)/lib64
CUDA_INCDIR   ?= $(CUDA_HOME)/include

ALIGNMENT ?= 4096 # Memory allocation alignment

#NODEBUG  = 1 # Disable debugging mode (use this for production releases)
#NOCUDA   = 1 # Disable CUDA support
#ANY_ARCH = 1 # Disable native architecture compilation
