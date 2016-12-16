
include config.mk
include user.mk

LIB_DIR = lib
INC_DIR = src
SRC_DIR = src

all:
	$(MAKE) -C $(SRC_DIR) all
.PHONY: all
test:
	$(MAKE) -C $(SRC_DIR) test
.PHONY: test
clean:
	$(MAKE) -C $(SRC_DIR) clean
.PHONY: clean
install: $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO_MAJ_MIN) $(INSTALL_INC_DIR)/$(BIFROST_NAME)
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

IMAGE_NAME ?= ledatelescope/bifrost
docker:
	docker build -t $(IMAGE_NAME):$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR) -t $(IMAGE_NAME) .
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
