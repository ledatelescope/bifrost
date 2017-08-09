
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

ifndef INSTALL_LIB_DIR
	INSTALL_LIB_DIR = /usr/local/lib
endif

ifndef INSTALL_INC_DIR
	INSTALL_INC_DIR = /usr/local/include
endif

BIFROST_NAME          = bifrost
LIBBIFROST_NAME       = lib$(BIFROST_NAME)
LIBBIFROST_MAJOR      = 0
LIBBIFROST_MINOR      = 8
LIBBIFROST_PATCH      = 0
LIBBIFROST_SO         = $(LIBBIFROST_NAME)$(SO_EXT)
LIBBIFROST_SO_MAJ     = $(LIBBIFROST_SO).$(LIBBIFROST_MAJOR)
LIBBIFROST_SO_MAJ_MIN = $(LIBBIFROST_SO_MAJ).$(LIBBIFROST_MINOR)
