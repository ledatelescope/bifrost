# Bifrost

| **`CPU/GPU Build`** | **`Coverage`** | 
|-----------------|----------------|
|[![GHA](https://github.com/ledatelescope/bifrost/actions/workflows/main.yml/badge.svg)](https://github.com/ledatelescope/bifrost/actions/workflows/main.yml) | [![Coverage Status](https://codecov.io/gh/ledatelescope/bifrost/branch/master/graph/badge.svg?token=f3ge1zWe5P)](https://codecov.io/gh/ledatelescope/bifrost) |

A stream processing framework for high-throughput applications.

### [![Paper](https://img.shields.io/badge/arXiv-1708.00720-blue.svg)](https://arxiv.org/abs/1708.00720)

### [Bifrost Documentation](http://ledatelescope.github.io/bifrost/)

See also the [Bifrost tutorial notebooks](tutorial/), which can be run
on Google Colab or any Jupyter environment where Bifrost is installed
(and a GPU is available).

### [Bifrost Roadmap](ROADMAP.md)

## A Simple Pipeline

Here's a snippet that reads Sigproc filterbank files, applies a
Fast Dispersion Measure Transform (FDMT) on the GPU, and writes
the results to a set of dedispersed time series files:

```python
import bifrost as bf
import sys

filenames = sys.argv[1:]

print "Building pipeline"
data = bf.blocks.read_sigproc(filenames, gulp_nframe=128)
data = bf.blocks.copy(data, 'cuda')
data = bf.blocks.transpose(data, ['pol', 'freq', 'time'])
data = bf.blocks.fdmt(data, max_dm=100.)
data = bf.blocks.copy(data, 'cuda_host')
bf.blocks.write_sigproc(data)

print "Running pipeline"
bf.get_default_pipeline().run()
print "All done"
```

## A More Complex Pipeline

Below is a longer snippet that demonstrates some additional features
of Bifrost pipelines, including the BlockChainer tool, block scopes,
CPU and GPU binding, data views, and dot graph output. This example
generates high-resolution spectra from Guppi Raw data:

```python
import bifrost as bf
import sys

filenames = sys.argv[1:]
f_avg = 4
n_int = 8

print "Building pipeline"
bc = bf.BlockChainer()
bc.blocks.read_guppi_raw(filenames, core=0)
bc.blocks.copy(space='cuda', core=1)
with bf.block_scope(fuse=True, gpu=0):
    bc.blocks.transpose(['time', 'pol', 'freq', 'fine_time'])
    bc.blocks.fft(axes='fine_time', axis_labels='fine_freq', apply_fftshift=True)
    bc.blocks.detect('stokes')
    bc.views.merge_axes('freq', 'fine_freq')
    bc.blocks.reduce('freq', f_avg)
    bc.blocks.accumulate(n_int)
bc.blocks.copy(space='cuda_host', core=2)
bc.blocks.write_sigproc(core=3)

pipeline = bf.get_default_pipeline()
print pipeline.dot_graph()
print "Running pipeline"
pipeline.shutdown_on_signals()
pipeline.run()
print "All done"
```

## Feature Overview

 - Designed for sustained high-throughput stream processing
 - Python API wraps fast C++/CUDA backend
 - Fast and flexible ring buffer specifically designed for processing continuous data streams
 - Native support for both system (CPU) and CUDA (GPU) memory spaces and computation
 - Fast kernels for transposition, dedispersion, correlation, beamforming and more
 - bfMap: JIT-compiled ND array transformations
 - Fast UDP data capture
 - A growing library of ready-to-use pipeline 'blocks'
 - Rich metadata enables seamless interoperability between blocks

## Installation

**For a quick demo which you can run in-browser without installation,
go to the following [link](https://colab.research.google.com/github/ledatelescope/bifrost/blob/master/BifrostDemo.ipynb).**

### C Dependencies

    $ sudo apt-get install exuberant-ctags

### Python Dependencies

 * numpy
 * contextlib2
 * pint
 * ctypesgen

```
$ sudo pip install numpy contextlib2 pint ctypesgen==1.0.2
```

### Bifrost Installation

To configure Bifrost for you your system and build the library, then run:

    $ ./configure
    $ make -j
    $ sudo make install

By default this will install the library and headers into /usr/local/lib and
/usr/local/include respectively.  You can use the --prefix option to configure
to change this.

You can call the following for a local Python installation:

    $ ./configure --with-pyinstall-flags=--user
    $ make -j
    $ sudo make install HAVE_PYTHON=0
    $ make -C python install

### Docker Container

Install dependencies:

 * [Docker Engine](https://docs.docker.com/engine/installation/)
 * [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

Build Docker image:

    $ make docker

Launch container:

    $ nvidia-docker run --rm -it ledatelescope/bifrost

For CPU-only builds:

    $ make docker-cpu
    $ docker run --rm -it ledatelescope/bifrost

### Running Tests

To run all CPU and GPU tests:

    $ make test

## Documentation

### [Online Bifrost Documentation](http://ledatelescope.github.io/bifrost/)

### Building the Docs with Docker

To quickly build the docs using Docker, ensure that you have
built a Bifrost container as `ledatelescope/bifrost`.
Then, inside the `docs` folder, execute `./docker_build_docs.sh`,
which will create a container called `bifrost_docs`, then
run it, and have it complete the docs-building process for you,
outputting the entire html documentation inside `docs/html` on
your machine.

### Building the Docs from Scratch

Install sphinx and breathe using pip, and also install Doxygen.

Doxygen documentation can be generated by running:

    $ make doc

This documentation can then be used in a Sphinx build
by running

    $ make html

inside the /docs directory.

## Telemetry

By default Bifrost installs with basic Python telemetry enabled in
order to help inform how the software is used and to help inform future 
development.  The data collected as part of this consist seven things:
 * a timestamp for when the report is generated,
 * a unique installation identifier,
 * the Bifrost version being used, 
 * the execution time of the Python process that imports Bifrost,
 * which Bifrost modules are imported,
 * which Bifrost functions are used and their average execution times, and
 * which Bifrost scripts are used.
These data are sent to the Bifrost developers using a HTTP POST request where
they are aggregated.

Users can opt out of telemetry collection using:

```
python -m bifrost.telemetry --disable
```

This command will set a disk-based flag that disables the reporting process.

## Acknowledgement

If you make use of Bifrost as part of your data collection or analysis please
include the following acknowledgement in your publications:

> This research has made use of Bifrost (Cranmer et al. 2017).  Continued
> development of Bifrost is supported by NSF award OAC/2103707.

and cite:

> \bibitem[Cranmer et al.(2017)]{2017JAI.....650007C} Cranmer, M.~D., Barsdell, B.~R., Price, D.~C., et al.\ 2017, Journal of Astronomical Instrumentation, 6, 1750007. doi:10.1142/S2251171717500076

## Contributors

 * Ben Barsdell
 * Daniel Price
 * Miles Cranmer
 * Hugh Garsden
 * Jayce Dowell
