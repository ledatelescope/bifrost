
include config.mk
include user.mk

LIB_DIR = lib
INC_DIR = src
SRC_DIR = src

BIFROST_PYTHON_DIR = python

all: libbifrost python
.PHONY: all

libbifrost:
	$(MAKE) -C $(SRC_DIR) all
.PHONY: libbifrost

test:
	#$(MAKE) -C $(SRC_DIR) test
	cd test && ./download_test_data.sh ; python -m unittest discover
.PHONY: test
clean:
	$(MAKE) -C $(BIFROST_PYTHON_DIR) clean || true
	$(MAKE) -C $(SRC_DIR) clean
.PHONY: clean
install: $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO_MAJ_MIN) $(INSTALL_INC_DIR)/$(BIFROST_NAME)
	install -C -D -m 644 -t $(INSTALL_INC_DIR)/$(BIFROST_NAME)/config config.mk user.mk
	install -C -m 644 -t $(INSTALL_INC_DIR) \
		src/array_utils.hpp src/assert.hpp src/Complex.hpp src/cuda.hpp src/EnvVars.hpp \
		src/fft_kernels.h src/int_fastdiv.h src/Jones.hpp src/linalg_kernels.h src/ObjectCache.hpp \
		src/utils.hpp src/utils.hu src/Vector.hpp src/workspace.hpp src/ArrayIndexer.cuh src/IndexArray.cuh \
		src/ShapeIndexer.cuh
	$(MAKE) -C $(BIFROST_PYTHON_DIR) install
.PHONY: install
uninstall:
	rm -f $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO)
	rm -f $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO_MAJ)
	rm -f $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO_MAJ_MIN)
	rm -rf $(INSTALL_INC_DIR)/bifrost/
	rm -rf $(INSTALL_INC_DIR)/array_utils.hpp
	rm -rf $(INSTALL_INC_DIR)/assert.hpp
	rm -rf $(INSTALL_INC_DIR)/Complex.hpp
	rm -rf $(INSTALL_INC_DIR)/cuda.hpp
	rm -rf $(INSTALL_INC_DIR)/EnvVars.hpp
	rm -rf $(INSTALL_INC_DIR)/fft_kernels.h
	rm -rf $(INSTALL_INC_DIR)/int_fastdiv.h
	rm -rf $(INSTALL_INC_DIR)/Jones.hpp
	rm -rf $(INSTALL_INC_DIR)/linalg_kernels.h
	rm -rf $(INSTALL_INC_DIR)/ObjectCache.hpp
	rm -rf $(INSTALL_INC_DIR)/utils.hpp
	rm -rf $(INSTALL_INC_DIR)/utils.hu
	rm -rf $(INSTALL_INC_DIR)/Vector.hpp
	rm -rf $(INSTALL_INC_DIR)/workspace.hpp
	rm -rf $(INSTALL_INC_DIR)/ArrayIndexer.cuh
	rm -rf $(INSTALL_INC_DIR)/IndexArray.cuh
	rm -rf $(INSTALL_INC_DIR)/ShapeIndexer.cuh
	$(MAKE) -C $(BIFROST_PYTHON_DIR) uninstall
.PHONY: uninstall

doc: $(INC_DIR)/bifrost/*.h Doxyfile
	$(DOXYGEN) Doxyfile
.PHONY: doc

python: libbifrost
	$(MAKE) -C $(BIFROST_PYTHON_DIR) build
.PHONY: python

#GPU Docker build
IMAGE_NAME ?= ledatelescope/bifrost
docker:
	docker build --pull -t $(IMAGE_NAME):$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR) -f Dockerfile.gpu -t $(IMAGE_NAME) .
.PHONY: docker

#GPU Docker prereq build
# (To be used for testing new builds rapidly)
IMAGE_NAME ?= ledatelescope/bifrost
docker_prereq:
	docker build --pull -t $(IMAGE_NAME)_prereq:$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR) -f Dockerfile_prereq.gpu -t $(IMAGE_NAME)_prereq .
.PHONY: docker_prereq

#CPU-only Docker build
IMAGE_NAME ?= ledatelescope/bifrost
docker-cpu:
	docker build --pull -t $(IMAGE_NAME):$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR) -f Dockerfile.cpu -t $(IMAGE_NAME) .
.PHONY: docker

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
