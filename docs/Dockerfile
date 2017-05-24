# This container should provide all
#   pre-requisites for building Bifrost
#   documentation
FROM ledatelescope/bifrost

RUN pip install --no-cache-dir \
    git+https://github.com/sphinx-doc/sphinx@stable \
    git+https://github.com/michaeljones/breathe

RUN ["/bin/bash"]
