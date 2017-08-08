.DEFAULT_GOAL := all
SHELL := /bin/zsh
WITH_BIFROST := blocks/accumulate.py blocks/hdf.py bf_gpuspec.py
WITHOUT_BIFROST := guppi2spectra.c guppi2spectra_gpu.cu guppi2spectra_gpu.h filterbank.h

# Be silent
ifndef VERBOSE
.SILENT:
endif

download:
	WITH_BIFROST="$(WITH_BIFROST)" WITHOUT_BIFROST="$(WITHOUT_BIFROST)" ./download_files.sh

test:
	@echo
	wget -q https://raw.githubusercontent.com/MilesCranmer/vim-stream/master/vims -O vims; \
		chmod +x vims; \
		cd with_bifrost; \
		echo "With Bifrost:"; \
		echo "`lizard -t 12 -i 9999 $(WITH_BIFROST) | tail -n 4 | ../vims '1norm yyGp'`"; \
		cd ..; \
		cd without_bifrost/src; \
		echo "Without Bifrost:"; \
		echo "`lizard -t 12 -i 9999 $(WITHOUT_BIFROST) | tail -n 4 | ../../vims '1norm yyGp'`"; \
		cd ../..; \
		rm vims

all: test download
