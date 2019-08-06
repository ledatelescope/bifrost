
.SILENT:

include ../config.mk
include ../user.mk

LIBBIFROST_OBJS = \
  common.o \
  memory.o \
  affinity.o \
  cuda.o \
  ring.o \
  ring_impl.o \
  array.o \
  address.o \
  udp_socket.o \
  udp_capture.o \
  udp_transmit.o \
  unpack.o \
  quantize.o \
  proclog.o
ifndef NOCUDA
  # These files require the CUDA Toolkit to compile
  LIBBIFROST_OBJS += \
  transpose.o \
  fft.o \
  fft_kernels.o \
  fdmt.o \
  map.o \
  trace.o \
  linalg.o \
  linalg_kernels.o \
  reduce.o \
  fir.o \
  guantize.o \
  gunpack.o
endif

JIT_SOURCES ?= \
  Complex.hpp.jit \
  Vector.hpp.jit \
  IndexArray.cuh.jit \
  ArrayIndexer.cuh.jit \
  ShapeIndexer.cuh.jit \
  int_fastdiv.h.jit

MAKEFILES = ../config.mk ../user.mk Makefile

ifndef NOCUDA
# All CUDA archs supported by this version of nvcc
GPU_ARCHS_SUPPORTED := $(shell $(NVCC) -h | grep -Po "compute_[0-9]{2}" | cut -d_ -f2 | sort | uniq)
# Intersection of user-specified archs and supported archs
GPU_ARCHS_VALID     := $(shell echo $(GPU_ARCHS) $(GPU_ARCHS_SUPPORTED) | xargs -n1 | sort | uniq -d | xargs)
# Latest valid arch
GPU_ARCH_LATEST     := $(shell echo $(GPU_ARCHS_VALID) | rev | cut -d' ' -f1 | rev)

# This creates SASS for all valid requested archs, and PTX for the latest one
NVCC_GENCODE  ?= $(foreach arch, $(GPU_ARCHS_VALID), \
  -gencode arch=compute_$(arch),\"code=sm_$(arch)\") \
  -gencode arch=compute_$(GPU_ARCH_LATEST),\"code=compute_$(GPU_ARCH_LATEST)\"
endif

CXXFLAGS  += -std=c++11 -fPIC -fopenmp
NVCCFLAGS += -std=c++11 -Xcompiler "-fPIC" $(NVCC_GENCODE)

#NVCCFLAGS += -Xcudafe "--diag_suppress=unrecognized_gcc_pragma"
#NVCCFLAGS += --expt-relaxed-constexpr

ifndef NODEBUG
  CPPFLAGS  += -DBF_DEBUG=1
  CXXFLAGS  += -g
  NVCCFLAGS += -g
endif

LIB += -lgomp

ifdef TRACE
  CPPFLAGS   += -DBF_TRACE_ENABLED=1
endif

ifdef NUMA
  # Requires libnuma-dev to be installed
  LIB        += -lnuma
  CPPFLAGS   += -DBF_NUMA_ENABLED=1
endif

ifdef HWLOC
  # Requires libhwloc-dev to be installed
  LIB        += -lhwloc
  CPPFLAGS   += -DBF_HWLOC_ENABLED=1
endif

ifdef VMA
  # Requires Mellanox libvma to be installed
  LIB        += -lvma
  CPPFLAGS   += -DBF_VMA_ENABLED=1
endif

ifdef ALIGNMENT
  CPPFLAGS   += -DBF_ALIGNMENT=$(ALIGNMENT)
endif

ifdef CUDA_DEBUG
  NVCCFLAGS += -G
endif

ifndef NOCUDA
  CPPFLAGS  += -DBF_CUDA_ENABLED=1
  LIB       += -L$(CUDA_LIBDIR64) -L$(CUDA_LIBDIR) -lcuda -lcudart -lnvrtc -lcublas -lcudadevrt -L. -lcufft_static_pruned -lculibos -lnvToolsExt
endif

ifndef ANY_ARCH
  CXXFLAGS  += -march=native
  NVCCFLAGS += -Xcompiler "-march=native"
endif

LIB_DIR = ../lib
INC_DIR = .
CPPFLAGS += -I. -I$(INC_DIR) -I$(CUDA_INCDIR)

LIBBIFROST_VERSION_FILE = $(LIBBIFROST_NAME).version
LIBBIFROST_SO_STEM      = $(LIB_DIR)/$(LIBBIFROST_NAME)$(SO_EXT)
LIBBIFROST_SO           = $(LIBBIFROST_SO_STEM).$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR)
LIBBIFROST_SO_NAME      = $(LIBBIFROST_NAME)$(SO_EXT).$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR)

all: $(LIBBIFROST_SO)
.PHONY: all

$(LIBBIFROST_VERSION_FILE): $(INC_DIR)/bifrost/*.h
	$(CLEAR_LINE)
	@echo -n "Generating $(LIBBIFROST_VERSION_FILE)\r"
	ctags --version | grep -q "Exuberant" || {\
	  echo "*************************************" && \
	  echo "ERROR: Please install exuberant-ctags" && \
	  echo "*************************************" && \
	  false; }
	echo "VERS_$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR) {" > $@
	echo "  global:" >> $@
	ctags -x --c-kinds=p $^ | awk '{print "    " $$1 ";"}' >> $@
	echo "  local:"  >> $@
	echo "    *;"    >> $@
	echo "};"        >> $@

ifndef NOCUDA
# TODO: Need to deal with 32/64 detection here
LIBCUFFT_STATIC = $(CUDA_LIBDIR64)/libcufft_static.a
# All PTX archs included in the lib (typically only one)
CUFFT_PTX_ARCHS := $(shell cuobjdump --list-ptx $(LIBCUFFT_STATIC) | grep -Po "sm_[0-9]{2}" | cut -d_ -f2 | sort | uniq)
# Latest PTX arch included in the lib
CUFFT_PTX_LATEST_ARCH := $(shell echo $(CUFFT_PTX_ARCHS) | rev | cut -d' ' -f1 | rev)
CUFFT_STATIC_GENCODE = -gencode arch=compute_$(CUFFT_PTX_LATEST_ARCH),\"code=compute_$(CUFFT_PTX_LATEST_ARCH)\"
libcufft_static_pruned.a: $(LIBCUFFT_STATIC) Makefile
	# We prune out all archs except those to be included in libbifrost *and*
	#   the latest PTX arch included in libcufft_static.
	#   E.g., We may have GPU_ARCHS="35 61" but libcufft_static might only
	#     include sm_60 and compute_60, so we need to keep compute_60 in order
	#     to support sm_61.
	nvprune -o $@ $(NVCC_GENCODE) $(CUFFT_STATIC_GENCODE) $<
fft_kernels.o: fft_kernels.cu fft_kernels.h Makefile
	# Note: This needs to be compiled with "-dc" to make CUFFT callbacks work
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler "$(GCCFLAGS)" -dc $(OUTPUT_OPTION) $<
_cuda_device_link.o: Makefile fft_kernels.o libcufft_static_pruned.a
	$(CLEAR_LINE)
	@echo -n "Linking _cuda_device_link.o\r"
	# TODO: "nvcc -dlink ..." does not error or warn when a -lblah is not found
	@ls ./libcufft_static_pruned.a > /dev/null
	$(NVCC) -dlink -o $@ $(NVCCFLAGS) fft_kernels.o -L. -lcufft_static_pruned
CUDA_DEVICE_LINK_OBJ = _cuda_device_link.o
else
CUDA_DEVICE_LINK_OBJ =
endif

# Note: $(LIB) must go at after OBJS
$(LIBBIFROST_SO): $(LIBBIFROST_OBJS) $(LIBBIFROST_VERSION_FILE) $(CUDA_DEVICE_LINK_OBJ)
	$(CLEAR_LINE)
	@echo -n "Linking $(LIBBIFROST_SO_NAME)\r"
	mkdir -p $(LIB_DIR)
	$(LINKER) $(SHARED_FLAG) -Wl,--version-script=$(LIBBIFROST_VERSION_FILE),$(SONAME_FLAG),$(LIBBIFROST_NAME)$(SO_EXT).$(LIBBIFROST_MAJOR) -o $@ $(LIBBIFROST_OBJS) $(CUDA_DEVICE_LINK_OBJ) $(LIB) $(LDFLAGS)
	ln -s -f $(LIBBIFROST_SO_NAME) $(LIBBIFROST_SO_STEM).$(LIBBIFROST_MAJOR)
	ln -s -f $(LIBBIFROST_SO_NAME) $(LIBBIFROST_SO_STEM)
	$(CLEAR_LINE)
	@echo "Successfully built $(LIBBIFROST_SO_NAME)"

*.o: $(MAKEFILES)

map.o: $(JIT_SOURCES)

stringify: stringify.cpp
	g++ -o stringify -Wall -O3 stringify.cpp
%.jit: % stringify
	@$(CLEAR_LINE)
	@echo -n "Building JIT version of $<\r"
	./stringify $< > $@

clean:
	rm -f *.o *.so *.dylib *.a test_bifrost
	rm -f $(LIBBIFROST_VERSION_FILE)
	rm -f $(LIBBIFROST_SO)
	rm -f $(LIBBIFROST_SO_STEM).$(LIBBIFROST_MAJOR)
	rm -f $(LIBBIFROST_SO_STEM)
	rm -f stringify
	rm -f $(DEPDIR)/*.d
	rm -f $(DEPDIR)/*.Td
	rmdir $(DEPDIR)
.PHONY: clean

include autodep.mk
