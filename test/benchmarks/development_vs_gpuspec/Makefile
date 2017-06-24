SHELL := /bin/bash

# Be silent
ifndef VERBOSE
.SILENT:
endif

download:
	mkdir -p with_bifrost
	[ -d "with_bifrost/.git" ] || git clone --depth=1 https://github.com/telegraphic/bunyip.git ./with_bifrost
	mkdir -p without_bifrost
	[ -d "without_bifrost/.git" ] || git clone --depth=1 https://github.com/UCBerkeleySETI/gbt_seti ./without_bifrost

test:
	cd with_bifrost; \
		lizard -w -t 12
	cd without_bifrost/src; \
		lizard -w -t 12

all: test download
