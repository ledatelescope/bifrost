
include config.mk
include user.mk

LIB_DIR = lib
INC_DIR = src
SRC_DIR = src

BIFROST_PYTHON_DIR = python
BIFROST_PYTHON_VERSION_FILE = $(BIFROST_PYTHON_DIR)/bifrost/version.py

all: libbifrost $(BIFROST_PYTHON_VERSION_FILE) python
.PHONY: all

libbifrost:
	$(MAKE) -C $(SRC_DIR) all
.PHONY: libbifrost

$(BIFROST_PYTHON_VERSION_FILE): config.mk
	@echo "__version__ = \"$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR).$(LIBBIFROST_PATCH)\"" > $@

test:
	#$(MAKE) -C $(SRC_DIR) test
	cd test && python -m unittest discover
.PHONY: test
clean:
	$(MAKE) -C $(BIFROST_PYTHON_DIR) clean || true
	$(MAKE) -C $(SRC_DIR) clean
	rm -f $(BIFROST_PYTHON_VERSION_FILE)
.PHONY: clean
install: $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO_MAJ_MIN) $(INSTALL_INC_DIR)/$(BIFROST_NAME)
	$(MAKE) -C $(BIFROST_PYTHON_DIR) install
.PHONY: install
uninstall:
	rm -f $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO)
	rm -f $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO_MAJ)
	rm -f $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO_MAJ_MIN)
	rm -rf $(INSTALL_INC_DIR)/bifrost/
.PHONY: uninstall

doc: $(INC_DIR)/bifrost/*.h Doxyfile
	$(DOXYGEN) Doxyfile
.PHONY: doc

python: libbifrost
	$(MAKE) -C $(BIFROST_PYTHON_DIR) build
.PHONY: python

#GPU Docker build
docker-base:
	echo "FROM $(CUDA_IMAGE_NAME)" > _Dockerfile.tmp
	cat Dockerfile.base >> _Dockerfile.tmp
	docker build --pull -t $(IMAGE_NAME):latest-base -t $(IMAGE_NAME):$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR)-base -f _Dockerfile.tmp .
	rm _Dockerfile.tmp
.PHONY: docker-base

docker: docker-base
	echo "FROM $(IMAGE_NAME):latest-base" > _Dockerfile.tmp
	cat Dockerfile >> _Dockerfile.tmp
	docker build --build-arg make_args="" -t $(IMAGE_NAME):latest -t $(IMAGE_NAME):$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR) -f _Dockerfile.tmp .
	rm _Dockerfile.tmp
.PHONY: docker

#CPU-only Docker build
docker-base-cpu:
	echo "FROM $(CPU_IMAGE_NAME)" > _Dockerfile.tmp
	cat Dockerfile.base >> _Dockerfile.tmp
	docker build --pull -t $(IMAGE_NAME):latest-base-cpu -t $(IMAGE_NAME):$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR)-base-cpu -f _Dockerfile.tmp .
	rm _Dockerfile.tmp
.PHONY: docker-base

docker-cpu: docker-base-cpu
	echo "FROM $(IMAGE_NAME):latest-base-cpu" > _Dockerfile.tmp
	cat Dockerfile >> _Dockerfile.tmp
	docker build --build-arg make_args="NOCUDA=1" -t $(IMAGE_NAME):latest-cpu -t $(IMAGE_NAME):$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR)-cpu -f _Dockerfile.tmp .
	rm _Dockerfile.tmp
.PHONY: docker-cpu

# TODO: Consider adding a mode 'develop=1' that makes symlinks instead of copying
#         the library and headers.

DRY_RUN ?= 0

$(INSTALL_LIB_DIR)/$(LIBBIFROST_SO_MAJ_MIN): $(LIB_DIR)/$(LIBBIFROST_SO_MAJ_MIN)
ifeq ($(DRY_RUN),0)
	cp $< $@
	ln -f -s $(LIBBIFROST_SO_MAJ_MIN) $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO).$(LIBBIFROST_MAJOR)
	ln -f -s $(LIBBIFROST_SO_MAJ_MIN) $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO)
else
	@echo "cp $< $@"
	@echo "ln -f -s $(LIBBIFROST_SO_MAJ_MIN) $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO).$(LIBBIFROST_MAJOR)"
	@echo "ln -f -s $(LIBBIFROST_SO_MAJ_MIN) $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO)"
endif

$(INSTALL_INC_DIR)/bifrost: $(INC_DIR)/bifrost/*.h #$(INC_DIR)/bifrost/*.hpp
ifeq ($(DRY_RUN),0)
	mkdir -p $@
	cp $? $@/
else
	@echo "mkdir -p $@"
	@echo "cp $? $@/"
endif
