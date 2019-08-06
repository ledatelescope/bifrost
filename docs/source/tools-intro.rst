Monitoring Tools
================

Bifrost provides some command-line tools for monitoring the performance of running pipelines.
In the ``tools/`` directory there are 

*(aka seeing C in Python pipelines)*

Bifrost is based on two maxims: 1) by itself, C sucks. 2) by itself,
Python sucks.

Bifrost bridges the two languages using some clever python libraries, a
customized numpy array class, and enforcing some pipeline friendly
design choices.

NVIDIA Profiler
---------------

The NVIDIA Profiler and Visual Profiler tools (part of the CUDA Toolkit)
can be used to profile and visualize Bifrost pipelines. Applications can
be launched directly from the Visual Profiler (nvvp), or a profile can
first be generated using the nvprof command line tool:

.. code::

    $ nvprof -o my_pipeline.nvprof python my_pipeline.py

The generated .nvprof file can then be imported into the Visual Profiler
for visualisation and analysis.

To obtain a more detailed profile of pipeline execution, rebuild the bifrost library
with the setting TRACE=1 (either by changing ``user.mk`` or by passing it as an
argument to the ``make`` command).


Pipeline in /dev/shm
--------------------

Details about the currently running bifrost pipeline are available in the ``/dev/shm`` directory.
They are mapped into a directory structure (use the linux ``tree`` utility to view it):

.. code::

    dancpr@bldcpr:/bldata/bifrost/tools$ tree /dev/shm/bifrost
    /dev/shm/bifrost
    └── 17263
        └── Pipeline_0
            ├── AccumulateBlock_0
            │   ├── bind
            │   ├── in
            │   ├── out
            │   ├── perf
            │   └── sequence0
            ├── BlockScope_1
            │   ├── PrintHeaderBlock_0
            │   │   ├── bind
            │   │   ├── in
            │   │   ├── out
            │   │   ├── perf
            │   │   └── sequence0
            │   └── TransposeBlock_0
            │       ├── bind
            │       ├── in
            │       ├── out
            │       ├── perf
            │       └── sequence0
            ├── BlockScope_13
            ├...

like_top.py
-----------

The main performance monitoring tools is ``like_top.py``. This is, as the name suggests, like the linux utility ``top``.


..code::

    like_top.py - bldcpr - load average: 0.59, 0.14, 0.05
    Processes: 516 total, 1 running
    CPU(s):  1.9%us,  1.4%sy,  0.0%ni, 84.5%id, 12.1%wa,  0.0%hi,  0.0%si,  0.0%st
    Mem:   32341840k total,  19834116k used,  12507724k free,    515556k buffers
    Swap:  32938492k total,    767408k used,  32171084k free,  17982316k cached

       PID            Block  Core   %CPU    Total  Acquire  Process  Reserve  Cmd
     19154  GuppiRawSourceB     0    9.4    0.714    0.000    0.714    0.000  python ./bf_gpuspec_midres.py ../pulsa
     19154       FftBlock_0     3    4.4    0.733    0.699    0.034    0.000  python ./bf_gpuspec_midres.py ../pulsa
     19154      CopyBlock_0     2    4.4    0.722    0.700    0.021    0.000  python ./bf_gpuspec_midres.py ../pulsa
     19154  TransposeBlock_     1    3.5    0.710    0.695    0.015    0.000  python ./bf_gpuspec_midres.py ../pulsa
     19154  HdfWriteBlock_0     6    0.4    3.220    3.213    0.007    0.000  python ./bf_gpuspec_midres.py ../pulsa
     19154    DetectBlock_0     4    1.0    0.738    0.733    0.005    0.000  python ./bf_gpuspec_midres.py ../pulsa
     19154  FftShiftBlock_0     3    4.4    0.738    0.734    0.005    0.000  python ./bf_gpuspec_midres.py ../pulsa
     19154      CopyBlock_1     6    0.4    2.816    2.813    0.003    0.000  python ./bf_gpuspec_midres.py ../pulsa
     19154  AccumulateBlock     5    4.0    0.005    0.005    0.001    0.000  python ./bf_gpuspec_midres.py ../pulsa
     19154  PrintHeaderBloc    -1           3.220    3.220    0.000    0.000  python ./bf_gpuspec_midres.py ../pulsa

* Acquire is the time spent waiting for input (i.e., waiting on upstream blocks), 
* Process is the time spent processing data, and 
* Reserve is the time spent waiting for output space to become available in the ring (i.e., waiting for downstream blocks).

Note: The CPU fraction will probably be 100% on any GPU block because it's currently set to spin (busy loop) while waiting for the GPU.

