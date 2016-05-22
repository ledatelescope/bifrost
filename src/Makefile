
include ../config.mk

NVCC_GENCODE  ?= $(foreach arch, $(GPU_ARCHS), \
  -gencode arch=compute_$(arch),\"code=sm_$(arch),compute_$(arch)\")
CXX_FLAGS  += -std=c++11 -fPIC -fopenmp
NVCC_FLAGS += -std=c++11 -Xcompiler "-fPIC" $(NVCC_GENCODE)

LIB_DIR = ../lib
INC_DIR = .

ifndef OS
  OS := $(shell uname -s)
endif

ifeq ($(OS),Linux)
  SO_EXT      = .so
  SHARED_FLAG = -shared
  SONAME_FLAG = -soname
else ifeq ($(OS),Darwin)
  SO_EXT = .dylib
  SHARED_FLAG = -dynamiclib
  SONAME_FLAG = -install_name
#else ifeq ($(OS),Windows_NT)
#  SO_EXT = .dll
else
  $(error Unsupported OS)
endif

INC = -I. -I$(INC_DIR) -I$(CUDA_INCDIR)
LIB = -lgomp

#ifdef DEBUG
#CXX_FLAGS  += -g -DBF_DEBUG=1
#NVCC_FLAGS += -g -DBF_DEBUG=1
#endif
ifndef NODEBUG
CXX_FLAGS  += -g -DBF_DEBUG=1
NVCC_FLAGS += -g -DBF_DEBUG=1
endif

#ifndef CUDA_DISABLED
ifndef NOCUDA
CXX_FLAGS  += -DBF_CUDA_ENABLED=1
NVCC_FLAGS += -DBF_CUDA_ENABLED=1
LIB        += -L$(CUDA_LIBDIR64) -L$(CUDA_LIBDIR) -lcudart
endif

ifdef ALIGNMENT
CXX_FLAGS  += -DBF_ALIGNMENT=$(ALIGNMENT)
NVCC_FLAGS += -DBF_ALIGNMENT=$(ALIGNMENT)
endif

ifndef ANY_ARCH
CXX_FLAGS  += -march=native
NVCC_FLAGS += -Xcompiler "-march=native"
endif

LIBBIFROST_OBJS = \
  common.o \
  memory.o \
  affinity.o \
  ring.o \
  ring_impl.o \
  #udp.o
ifndef NOCUDA
  # These files require CUDA to compile
  LIBBIFROST_OBJS += \
  transpose.o \
  #apsynth.o
endif
#LIBBIFROST_HEADERS = bfcommon.h bfmemory.h bfaffinity.h bfring.h bfring_impl.hpp bftranspose.h bfudp.h Socket.hpp
LIBBIFROST_NAME    = libbifrost
LIBBIFROST_MAJOR   = 0
LIBBIFROST_MINOR   = 6
LIBBIFROST_VERSION_FILE = libbifrost.version
LIBBIFROST_SO_STEM = $(LIB_DIR)/$(LIBBIFROST_NAME)$(SO_EXT)
LIBBIFROST_SO      = $(LIBBIFROST_SO_STEM).$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR)

#LIBBIFROST_HEADERS =  $(addprefix $(INC_DIR)/,$(LIBBIFROST_HEADERS))

all: $(LIBBIFROST_SO)
test: test_bifrost
	./test_bifrost

# Note: $(LIB) must go at after OBJS
#$(LIBBIFROST_NAME)$(SO_EXT).$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR)
$(LIBBIFROST_SO): $(LIBBIFROST_OBJS)
	mkdir -p $(LIB_DIR)
	$(LINKER) $(SHARED_FLAG) -Wl,--version-script=$(LIBBIFROST_VERSION_FILE),$(SONAME_FLAG),$(LIBBIFROST_NAME)$(SO_EXT).$(LIBBIFROST_MAJOR) -o $@ $(LIBBIFROST_OBJS) $(LIB) $(LINKER_FLAGS)
	ln -s -f $@ $(LIBBIFROST_SO_STEM).$(LIBBIFROST_MAJOR)
	ln -s -f $@ $(LIBBIFROST_SO_STEM)

# TODO: Should depend on all referenced headers (or just all headers)
%.o: %.cpp $(INC_DIR)/bifrost/%.h*
	$(CXX) -o $@	-c $(CXX_FLAGS) $(INC) $<
%.o: %.cpp  %.h*
	$(CXX) -o $@	-c $(CXX_FLAGS) $(INC) $<
%.o: %.cu
	$(NVCC) -o $@	-c $(NVCC_FLAGS) $(INC) $<
#bfring.o: bfring.cpp bfring.h
#	$(CXX) -o $@ -c $(CXX_FLAGS) $(INC) $<

test_bifrost: test_bifrost.cpp $(LIBBIFROST_SO) $(INC_DIR)/bifrost.hpp $(INC_DIR)/bifrost.h
	$(CXX) -o $@ $(CXX_FLAGS) $(INC) $< -L$(LIB_DIR) -lbifrost

clean:
	rm -f *.o *.so *.dylib *.a test_bifrost
	rm -f $(LIBBIFROST_SO)
	rm -f $(LIBBIFROST_SO_STEM).$(LIBBIFROST_MAJOR)
	rm -f $(LIBBIFROST_SO_STEM)
.PHONY: clean
