# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = bifrost
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile generate_python_reference generate_cpp_reference
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

generate_python_reference:
	sphinx-apidoc -o source -d 5 --force ../python/bifrost/
	rm source/modules.rst
	sed -i '1s/.*/Python Reference/' source/bifrost.rst
	sed -i '2s/.*/================/' source/bifrost.rst
	sed -i '1s/.*/Block Library Reference/' source/bifrost.blocks.rst
	sed -i '2s/.*/=======================/' source/bifrost.blocks.rst
.PHONY: generate_python_reference

generate_cpp_reference:
	breathe-apidoc -o source -p bifrost --force ./doxygen/xml/
	rm -rf source/file
.PHONY: generate_cpp_reference
