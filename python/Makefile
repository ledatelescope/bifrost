
include ../config.mk

INC_DIR = ../src

BIFROST_PYTHON_VERSION_FILE  = bifrost/version.py
BIFROST_PYTHON_BINDINGS_FILE = bifrost/libbifrost_generated.py

all: build
.PHONY: all

$(BIFROST_PYTHON_VERSION_FILE): ../config.mk
	@echo "__version__ = \"$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR).$(LIBBIFROST_PATCH)\"" > $@

$(BIFROST_PYTHON_BINDINGS_FILE): $(INC_DIR)/bifrost/*.h
	ctypesgen.py -l$(BIFROST_NAME) -I$(INC_DIR) $^ -o $@
	# WAR for 'const char**' being generated as POINTER(POINTER(c_char)) instead of POINTER(c_char_p)
	sed -i 's/POINTER(c_char)/c_char_p/g' $@
	# WAR for a buggy WAR in ctypesgen that breaks type checking and auto-byref functionality
	sed -i 's/def POINTER/def POINTER_not_used/' $@
	# WAR for a buggy WAR in ctypesgen that breaks string buffer arguments (e.g., as in address.py)
	sed -i 's/class String/String = c_char_p\nclass String_not_used/' $@
	sed -i 's/String.from_param/String_not_used.from_param/g' $@

build: bifrost/*.py $(BIFROST_PYTHON_VERSION_FILE) $(BIFROST_PYTHON_BINDINGS_FILE)
	python setup.py build $(PYBUILDFLAGS)
.PHONY: build

install: build
	python setup.py install $(PYINSTALLFLAGS)
.PHONY: install

clean:
	python setup.py clean --all
	rm -f $(BIFROST_PYTHON_VERSION_FILE)
	rm -f $(BIFROST_PYTHON_BINDINGS_FILE)
.PHONY: clean
