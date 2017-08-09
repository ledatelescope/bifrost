FROM ledatelescope/bifrost:cpu-base

MAINTAINER Ben Barsdell <benbarsdell@gmail.com>

ARG DEBIAN_FRONTEND=noninteractive

ENV TERM xterm

# Build the library
WORKDIR /bifrost
COPY . .
RUN make clean && \
    make -j NOCUDA=1 && \
    make doc && \
    make install

ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}

# IPython
EXPOSE 8888

WORKDIR /workspace
RUN ["/bin/bash"]
