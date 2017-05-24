# This container should provide all
#   pre-requisites for building Bifrost
#   documentation
FROM ledatelescope/bifrost

RUN pip install --no-cache-dir \
    git+https://github.com/sphinx-doc/sphinx@stable \
    git+https://github.com/michaeljones/breathe

WORKDIR "/bifrost"

RUN make doc && \
    cd docs && \
    /bin/bash generate_cpp_reference.sh && \
    /bin/bash generate_python_reference.sh && \
    make html

WORKDIR "/bifrost/docs/build/html"

RUN ["/bin/bash"]
