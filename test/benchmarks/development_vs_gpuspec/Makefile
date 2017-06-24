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
	@echo
	cd with_bifrost; \
		echo $$(lizard -t 12 | tail -n 3)
	cd without_bifrost/src; \
		echo $$(lizard -t 12 -i 9999 cufft.h guppi2spectra.c guppi2spectra_gpu.cu guppi2spectra_gpu.h filterbank.h | tail -n 3)

all: test download
