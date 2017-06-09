Bifrost Benchmarks
==================

This is a list of benchmark tests for Bifrost.

1. :code:`general/` - General performance benchmarks
    1. Pipelines
        1. Dedispersion pipeline
        #. GUPPI raw to filterbank
        #. Iterated least squares
    #. Blocks
        1. accumulate
        #. audio
        #. binary_io
        #. copy
        #. detect
        #. fdmt
        #. fft
        #. fftshift
        #. guppi_raw
        #. print_header
        #. quantize
        #. reverse
        #. scrunch
        #. sigproc
        #. transpose
        #. unpack
        #. wav
    #. CUDA kernel generation
#. :code:`performance_vs_serial/` - Performance comparisons with Serial
    1. :code:`linear_fft_pipeline.py` - Linear repeated-FFT pipeline
#. :code:`development_vs_serial/` - Development-effort comparisons with Serial
    1. Pipeline using existing blocks
    #. Pipeline using new blocks
    #. GPU pipeline using new blocks
#. :code:`development_vs_psrdada/` - Development-effort comparisons with PSRDADA
    1. Packet capture pipeline
        1. Source lines of code
        #. Source lines of code per function
        #. Cyclomatic complexity per function
    #. Overall codebase
        1. Source lines of code
        #. Source lines of code per function
        #. Cyclomatic complexity per function
#. :code:`development_function_specific` - Function-specific development-effort comparisons
    1. Element-wise CUDA kernel
    #. Non-element-wise CUDA kernel

.. #. Performance comparisons with PSRDADA
..     1. Packet capture pipeline
