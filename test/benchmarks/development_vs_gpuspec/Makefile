SHELL := /bin/zsh
WITH_BIFROST := blocks/accumulate.py blocks/hdf.py
WITHOUT_BIFROST := guppi2spectra.c guppi2spectra_gpu.cu guppi2spectra_gpu.h filterbank.h

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
	wget -q https://raw.githubusercontent.com/MilesCranmer/vim-stream/master/vims -O vims; \
		chmod +x vims; \
		cd with_bifrost; \
		echo "With Bifrost:"; \
		$$(cat bf_gpuspec_hires.py | ../vims -e 'class Hdf' 'V/^\S\<enter>kd' > bf_gpuspec_hires2.py); \
		echo "`lizard -t 12 -i 9999 bf_gpuspec_hires2.py $(WITH_BIFROST) | tail -n 4 | ../vims '1norm yyGp'`"; \
		cd ..; \
		cd without_bifrost/src; \
		echo "Without Bifrost:"; \
		echo "`lizard -t 12 -i 9999 $(WITHOUT_BIFROST) | tail -n 4 | ../../vims '1norm yyGp'`"
		rm vims

all: test download
