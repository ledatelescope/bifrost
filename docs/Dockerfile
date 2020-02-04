# This container builds the Bifrost documentation
# using the ledatelescope/bifrost container, and
# puts it into the folder /bifrost/docs/build/html
# inside a new container.
FROM ledatelescope/bifrost

# Install pre-requisite documentation libraries at
# a specific commit that is known to work
RUN pip install --no-cache-dir \
    git+https://github.com/sphinx-doc/sphinx.git@1cd87a11ef8df6b783d2e48d9668370a4e1640c9 \
    git+https://github.com/michaeljones/breathe.git@f8f952d25581ecb202bc2caa4bbe53d61c432655

WORKDIR "/bifrost"

# Build the docs
RUN make doc && \
    cd docs && \
    make html

WORKDIR "/bifrost/docs/build/html"

RUN ["/bin/bash"]
