ARG BASE_CONTAINER=nvidia/cuda:10.1-devel-ubuntu18.04
FROM $BASE_CONTAINER

LABEL maintainer="Jayce Dowell <jdowell@unm.edu>"
ARG LSL_USER="lwa"
ARG LSL_UID="1000"
ARG LSL_GID="100"

# Build-time metadata as defined at http://label-schema.org
ARG BUILD_DATE
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="Bifrost - Jupyter image" \
      org.label-schema.description="Image with CUDA, Bifrost, LSL, and a useful Jupyter stack" \
      org.label-schema.url="https://github.com/ledatelescope/bifrost" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/ledatelescope/bifrost_tutorial" \
      org.label-schema.schema-version="1.0"

USER root

# Install all OS dependencies for fully functional notebook server
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        doxygen \
        emacs-nox \
        exuberant-ctags \
        fonts-liberation \
        git \
        inkscape \
        jed \
        libboost-python-dev \
        libcfitsio-dev \
        libfftw3-dev \
        libgdbm-dev \
        libhdf5-dev \
        libhwloc-dev \
        libnuma-dev \
        libsm6 \
        libxext-dev \
        libxrender1 \
        locales \
        lmodern \
        nano \
        netcat \
        pkg-config \
        python3-dev \
        python3-pip \
        python3-setuptools \
        run-one \
        software-properties-common \
        sudo \
        swig \
        tcsh \
        vim \
        wget \
        # ---- nbconvert dependencies ----
        texlive-xetex \
        texlive-fonts-recommended \
        texlive-generic-recommended \
        # Optional dependency
        texlive-fonts-extra \
        # ----
        tzdata \
        unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Configure the environment
ENV SHELL=/bin/bash \
    LSL_USER=$LSL_USER \
    LSL_UID=$LSL_UID \
    LSL_GID=$LSL_GID \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV HOME=/home/$LSL_USER
RUN python3 -m pip install --upgrade virtualenv

# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

# Enable prompt color in the skeleton .bashrc before creating the default LSL_USER
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc

# Create LSL_USER with name lwa user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su && \
    sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers && \
    sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && \
    useradd -m -s /bin/bash -N -u $LSL_UID $LSL_USER && \
    chmod g+w /etc/passwd && \
    fix-permissions $HOME

USER $LSL_UID
WORKDIR $HOME

# Activate the environment
ENV VIRTUAL_ENV=$HOME/venv
RUN python3 -m virtualenv -p python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Setup LSL
RUN pip install \
        setuptools \
        numpy \
        matplotlib \
        scipy \
        h5py \
        pyephem==3.7.6.0 && \
    pip install aipy && \
    pip install lsl
    
# Setup Bifrost
## Dependencies + numba
RUN pip install \
        contextlib2 \
        graphviz \
        numba \
        pint \
        simplejson \
        ctypesgen==1.0.2
## Bifrost
RUN git clone https://github.com/ledatelescope/bifrost.git && \
    cd /home/$LSL_USER/bifrost
RUN cd /home/$LSL_USER/bifrost && \
    ./configure --with-gpu-archs="35 50 61 75" && \
    make -j all && \
    make doc && \
    cd /home/$LSL_USER && \
    ln -s /home/$LSL_USER/bifrost/tutorial /home/$LSL_USER/bifrost_tutorial

# Back to root
USER root

RUN cd /home/$LSL_USER/bifrost && \
    make install

# Setup work directory for backward-compatibility
RUN mkdir /home/$LSL_USER/work && \
    fix-permissions /home/$LSL_USER

# Install Tini
RUN pip install tini && \
    fix-permissions /home/$LSL_USER

# Setup Jupyter
RUN pip install \
        jupyterlab \
        jupyterhub \
        jupyter_client \
        nbformat \
        nbconvert && \
    jupyter notebook --generate-config && \
    rm -rf /home/$LSL_USER/.cache/yarn && \
    fix-permissions /home/$LSL_USER

EXPOSE 8888
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/lwa/venv/lib"

# Configure container startup
# Add Tini
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]
CMD ["start-notebook.sh"]

# Copy local files as late as possible to avoid cache busting
COPY start.sh start-notebook.sh start-singleuser.sh /usr/local/bin/
RUN chmod a+rx /usr/local/bin/start*.sh
COPY jupyter_notebook_config.py /etc/jupyter/

# Fix permissions on /etc/jupyter as root
USER root
RUN fix-permissions /etc/jupyter/

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME /home/$LSL_USER/.cache/
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
    fix-permissions /home/$LSL_USER

# Switch back to lwa to avoid accidental container runs as root
USER $LSL_UID
