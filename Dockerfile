# Note: FROM command is inserted by Makefile
# TODO: As of Docker-CE 17.05, we could use this better approach instead:
#ARG base_layer
#FROM $base_layer

MAINTAINER Ben Barsdell <benbarsdell@gmail.com>

ARG make_args

# Build the library
WORKDIR /bifrost
COPY . .
RUN make clean && \
    make -j16 $make_args && \
    make doc && \
    make install

WORKDIR /workspace
RUN ["/bin/bash"]
