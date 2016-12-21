
LIBBIFROST_OBJS = \
  common.o \
  memory.o \
  affinity.o \
  ring.o \
  ring_impl.o \
  array.o \
  address.o \
  udp_socket.o \
  #unpack.o \
  udp_capture.o
ifndef NOCUDA
  # These files require the CUDA Toolkit to compile
  LIBBIFROST_OBJS += \
  cuda.o \
  transpose.o \
  fft.o \
  fdmt.o \
  #correlate.o \
  blas.o \
  transform.o
  #apsynth.o
endif

.SILENT:

include ../config.mk
include ../user.mk

MAKEFILES = ../config.mk ../user.mk Makefile

NVCC_GENCODE  ?= $(foreach arch, $(GPU_ARCHS), \
  -gencode arch=compute_$(arch),\"code=sm_$(arch),compute_$(arch)\")

CXXFLAGS  += -std=c++11 -fPIC -fopenmp
NVCCFLAGS += -std=c++11 -Xcompiler "-fPIC" $(NVCC_GENCODE)

ifndef NODEBUG
  CXXFLAGS  += -g -DBF_DEBUG=1
  NVCCFLAGS += -g -DBF_DEBUG=1
endif

LIB += -lgomp

ifdef HWLOC
  LIB        += -lhwloc
  CXXFLAGS   += -DBF_HWLOC_ENABLED=1
endif

ifdef ALIGNMENT
  CXXFLAGS   += -DBF_ALIGNMENT=$(ALIGNMENT)
  NVCCFLAGS  += -DBF_ALIGNMENT=$(ALIGNMENT)
endif

ifdef CUDA_DEBUG
  NVCCFLAGS += -G
endif

ifndef NOCUDA
  CXXFLAGS  += -DBF_CUDA_ENABLED=1
  NVCCFLAGS += -DBF_CUDA_ENABLED=1
  LIB       += -L$(CUDA_LIBDIR64) -L$(CUDA_LIBDIR) -lcuda -lcudart -lnvrtc -lcublas -lcufft
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

# Note: $(LIB) must go at after OBJS
$(LIBBIFROST_SO): $(LIBBIFROST_OBJS) $(LIBBIFROST_VERSION_FILE)
	$(CLEAR_LINE)
	@echo -n "Linking $(LIBBIFROST_SO_NAME)\r"
	mkdir -p $(LIB_DIR)
	$(LINKER) $(SHARED_FLAG) -Wl,--version-script=$(LIBBIFROST_VERSION_FILE),$(SONAME_FLAG),$(LIBBIFROST_NAME)$(SO_EXT).$(LIBBIFROST_MAJOR) -o $@ $(LIBBIFROST_OBJS) $(LIB) $(LDFLAGS)
	ln -s -f $(LIBBIFROST_SO_NAME) $(LIBBIFROST_SO_STEM).$(LIBBIFROST_MAJOR)
	ln -s -f $(LIBBIFROST_SO_NAME) $(LIBBIFROST_SO_STEM)
	$(CLEAR_LINE)
	@echo "Successfully built $(LIBBIFROST_SO_NAME)"

*.o: $(MAKEFILES)

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
