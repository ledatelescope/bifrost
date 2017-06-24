SHELL := /bin/bash

# Be silent
ifndef VERBOSE
.SILENT:
endif

download:
	mkdir -p with_bifrost
	[ -d "with_bifrost/.git" ] || git clone --depth=1 https://github.com/telegraphic/bunyip.git ./with_bifrost
	mkdir -p without_bifrost

all: download
