
include config.mk

LIB_DIR = lib
INC_DIR = src
INSTALL_LIBBIFROST_SO_FILE = $(INSTALL_LIB_DIR)/$(LIBBIFROST_SO)

all:
	$(MAKE) -C ./src all
.PHONY: all
test:
	$(MAKE) -C ./src test
.PHONY: test
clean:
	$(MAKE) -C ./src clean
.PHONY: clean
install: all $(INSTALL_LIBBIFROST_SO_FILE) $(INSTALL_INC_DIR)/$(BIFROST_NAME)
.PHONY: install
uninstall:
	rm -f $(INSTALL_LIBBIFROST_SO_FILE)
	rm -f $(INSTALL_LIBBIFROST_SO_FILE).$(LIBBIFROST_MAJOR)
	rm -f $(INSTALL_LIBBIFROST_SO_FILE).$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR)
.PHONY: uninstall

$(INSTALL_LIB_DIR)/%$(SO_EXT): $(LIB_DIR)/%$(SO_EXT)
	cp $< $@
	ln -f -s $@ $@.$(LIBBIFROST_MAJOR)
	ln -f -s $@ $@.$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR)

$(INSTALL_INC_DIR)/$(BIFROST_NAME): $(INC_DIR)/$(BIFROST_NAME)
	mkdir -p $@
	cp $</*.h $@
