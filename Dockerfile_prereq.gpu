FROM nvidia/cuda:8.0

MAINTAINER Ben Barsdell <benbarsdell@gmail.com>

ARG DEBIAN_FRONTEND=noninteractive

# Get dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        pkg-config \
        software-properties-common \
        python \
        python-dev \
        doxygen \
        exuberant-ctags \
        nano \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
RUN pip --no-cache-dir install \
        setuptools \
        numpy \
        matplotlib \
        contextlib2 \
        simplejson \
        pint \
        graphviz

RUN git clone https://github.com/MatthieuDartiailh/pyclibrary.git && \
    cd pyclibrary && \
    python setup.py install

ENV TERM xterm

# Build the library
WORKDIR /bifrost

ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}

# IPython
EXPOSE 8888

RUN ["/bin/bash"]
