
include ../config.mk

INC_DIR = ../src

BIFROST_PYTHON_VERSION_FILE  = bifrost/version.py
BIFROST_PYTHON_BINDINGS_FILE = bifrost/libbifrost_generated.py

PSRHOME ?= /usr/local
PSRDADA_PYTHON_BINDINGS_FILE = bifrost/libpsrdada_generated.py
PSRDADA_HEADERS = \
	$(PSRHOME)/include/dada_hdu.h \
	$(PSRHOME)/include/ipcio.h \
	$(PSRHOME)/include/ipcbuf.h \
	$(PSRHOME)/include/multilog.h

all: build
.PHONY: all

$(BIFROST_PYTHON_VERSION_FILE): ../config.mk
	@echo "__version__ = \"$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR).$(LIBBIFROST_PATCH)\"" > $@

define run_ctypesgen
	ctypesgen.py -l$1 -I$2 $^ -o $@
	# WAR for 'const char**' being generated as POINTER(POINTER(c_char)) instead of POINTER(c_char_p)
	sed -i 's/POINTER(c_char)/c_char_p/g' $@
	# WAR for a buggy WAR in ctypesgen that breaks type checking and auto-byref functionality
	sed -i 's/def POINTER/def POINTER_not_used/' $@
	# WAR for a buggy WAR in ctypesgen that breaks string buffer arguments (e.g., as in address.py)
	sed -i 's/class String/String = c_char_p\nclass String_not_used/' $@
	sed -i 's/String.from_param/String_not_used.from_param/g' $@
	sed -i 's/def ReturnString/ReturnString = c_char_p\ndef ReturnString_not_used/' $@
	sed -i '/errcheck = ReturnString/s/^/#/' $@
endef

ifeq "$(wildcard $(PSRDADA_HEADERS))" ""
PSRDADA_PYTHON_BINDINGS_FILE =
endif

$(PSRDADA_PYTHON_BINDINGS_FILE): $(PSRDADA_HEADERS)
	$(call run_ctypesgen,psrdada,$(PSRHOME)/include)
	# WAR for psrdada API using char* instead of void* for buffer pointers, which
	#   otherwise get inadvertently converted to Python strings.
	sed -i 's/c_char_p/POINTER(c_char)/g' $@

$(BIFROST_PYTHON_BINDINGS_FILE): $(INC_DIR)/bifrost/*.h
	$(call run_ctypesgen,$(BIFROST_NAME),$(INC_DIR))

build: bifrost/*.py Makefile $(BIFROST_PYTHON_VERSION_FILE) $(BIFROST_PYTHON_BINDINGS_FILE) $(PSRDADA_PYTHON_BINDINGS_FILE)
	python setup.py build $(PYBUILDFLAGS)
.PHONY: build

install: build
	python setup.py install $(PYINSTALLFLAGS)
.PHONY: install

clean:
	python setup.py clean --all
	rm -f $(BIFROST_PYTHON_VERSION_FILE)
	rm -f $(BIFROST_PYTHON_BINDINGS_FILE)
	rm -f $(PSRDADA_PYTHON_BINDINGS_FILE)
.PHONY: clean

uninstall:
	pip uninstall bifrost
