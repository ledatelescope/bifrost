
build: bifrost/*.py
	python setup.py build $(PYBUILDFLAGS)
.PHONY: build

install: build
	python setup.py install $(PYINSTALLFLAGS)
.PHONY: install

clean:
	python setup.py clean --all
.PHONY: clean
