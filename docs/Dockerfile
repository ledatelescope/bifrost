# This container builds the Bifrost documentation
# using the ledatelescope/bifrost container, and
# puts it into the folder /bifrost/docs/build/html
# inside a new container.
FROM ledatelescope/bifrost

# Install pre-requisite documentation libraries at
# a specific commit that is known to work
RUN pip install --no-cache-dir \
    git+https://github.com/sphinx-doc/sphinx.git@8aafa50af47f7536979a8c321461ec837c91c8e6 \
    git+https://github.com/michaeljones/breathe.git@d3eae7fac4d2ead062070fd149ec8bf839f74ed5

WORKDIR "/bifrost"

# Build the docs
RUN make doc && \
    cd docs && \
    make html

WORKDIR "/bifrost/docs/build/html"

RUN ["/bin/bash"]
