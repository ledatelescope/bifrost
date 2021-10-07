AC_INIT([bifrost], [0.9.0])
AC_LANG(C++)
AC_CONFIG_SRCDIR([src/cuda.cpp])

: ${CXXFLAGS="-O3 -Wall -pedantic"}

AC_PROG_CXX
AC_REQUIRE_CPP
AC_PROG_RANLIB

AC_ARG_ENABLE([cuda],
              [AS_HELP_STRING([--enable-cuda],
                              [enable cuda support (default=yes)])],
              [],
              [enable_cuda=yes])
AS_IF([test "x$enable_cuda" != xno],
      [AX_WITH_PROG(NVCC, nvcc)
       AX_WITH_PROG(NVPRUNE, nvprune)
       AX_WITH_PROG(CUOBJDUMP, cuobjdump)
       AC_SUBST([HAVE_CUDA], [1])],
      [])
AC_ARG_WITH([nvcc_flags],
            [AS_HELP_STRING([--with-nvcc-flags],
                            [flags to pass to NVCC (default='-O3 -Xcompiler -Wall')])],
            [],
            [with_nvcc_flags='-O3 -Xcompiler -Wall'])
AC_SUBST(NVCCFLAGS, $with_nvcc_flags)

AX_WITH_PROG(AWK, awk)

AX_WITH_PROG(CTAGS, ctags)
AS_IF([test "x${CTAGS}" = x],
      [AC_MSG_ERROR([Required program ctags was not found])],
      [])
AC_MSG_CHECKING([whether ${CTAGS} is exuberant])
AS_IF([! ${CTAGS} --version | grep -q Exuberant],
      [AC_MSG_RESULT([no])
       AC_MSG_ERROR([exhuberant ctags is required, but ${CTAGS} is a different version])],
      [AC_MSG_RESULT([yes])])

AX_OPENMP


AC_ARG_ENABLE([numa],
              AS_HELP_STRING([--enable-numa],
                             [enable NUMA support (default=yes)]),
              [],
              [enable_numa=yes])
AS_IF([test "x$enable_numa" != xno],
      [AC_CHECK_LIB([numa], [numa_tonode_memory],
                    [AC_SUBST([NUMA_LIB], [-lnuma])
                     AC_SUBST([NUMA_CPPFLAGS], [-DBF_NUMA_ENABLED=1])],
                    [])],
      [])

AC_ARG_ENABLE([hwloc],
              AS_HELP_STRING([--enable-hwloc],
                             [enable hwloc support (default=yes)]),
              [],
              [enable_hwloc=yes])
AS_IF([test "x$enable_hwloc" != xno],
      [AC_CHECK_LIB([hwloc], [hwloc_topology_init],
                    [AC_SUBST([HWLOC_LIB], [-lhwloc])
                     AC_SUBST([HWLOC_CPPFLAGS], [-DBF_HWLOC_ENABLED=1])],
                    [])],
      [])

AC_ARG_ENABLE([vma],
              AS_HELP_STRING([--enable-vma],
                             [enable vma support (default=no)]),
              [],
              [enable_vma=no])
AS_IF([test "x$enable_vma" != xno],
      [AC_CHECK_LIB([vma], [recvfrom_zcopy],
                    [AC_SUBST([VMA_LIB], [-lvma])
                     AC_SUBST([VMA_CPPFLAGS], [-DBF_VMA_ENABLED=1])],
                    [])],
      [])

AC_ARG_WITH([gpu_archs],
            [AS_HELP_STRING([--with-gpu-archs=...],
                            [default GPU architectures (default='35 61')])],
            [],
            [with_gpu_archs='35 61'])
AC_SUBST(GPU_ARCHS, $with_gpu_archs)

AC_ARG_WITH([shared_mem],
            [AS_HELP_STRING([--with-shared-mem=N],
                            [default GPU shared memory in bytes (default=16384)])],
            [],
            [with_shared_mem=16384])
AC_SUBST(GPU_SHAREDMEM, $with_shared_mem)

AC_ARG_WITH([alignment],
            [AS_HELP_STRING([--with-alignment=N],
                            [default memory alignment in bytes (default=4096)])],
            [],
            [with_alignment=4096])
AC_SUBST(ALIGNMENT, $with_alignment)

AC_ARG_ENABLE([debug],
              [AS_HELP_STRING([--enable-debug],
                              [enable debugging mode (default=no)])],
              [],
              [enable_debug=no])
AS_IF([test "x$enable_debug" != xno],
      [AC_SUBST(DEBUG_CPPFLAGS, [-DBF_DEBUG=1])
       AC_SUBST(DEBUG_CXXFLAGS, [-g])
       AC_SUBST(DEBUG_NVCCFLAGS, [-g])],
      [])

AC_ARG_ENABLE([trace],
              [AS_HELP_STRING([--enable-trace],
                              [enable tracing mode for nvprof/nvvp (default=no)])],
              [],
              [enable_trace=no])
AS_IF([test "x$enable_trace" != xno],
      [AC_SUBST(TRACE_CPPFLAGS, [-DBF_TRACE_ENABLED=1])],
      [])

AC_ARG_ENABLE([native_arch],
              [AS_HELP_STRING([--enable-native-arch],
                              [enable native architecture compilation (default=yes)])],
              [],
              [enable_native_arch=yes])
AS_IF([test "x$enable_native_arch" != xno],
      [AC_SUBST(ARCH_CXXFLAGS, [-march=native])
       AC_SUBST(ARCH_NVCCFLAGS, [-Xcompiler -march=native])],
      [])

AC_ARG_ENABLE([cuda_debug],
              [AS_HELP_STRING([--enable-cuda-debug],
                              [enable CUDA debugging (nvcc -G; default=no)])],
              [],
              [enable_cuda_debug=no])
AS_IF([test "x$enable_cuda_debug" != xno],
      [AC_SUBST(DEBUG_NVCCFLAGS, [-G])],
      [])

AX_PYTHON

AC_ARG_WITH([pybuild_flags],
            [AS_HELP_STRING([--with-pybuild-flags],
                            [build flags for python (default='')])],
            [],
            [])
AC_SUBST(PYBUILDFLAGS, $with_pybuild_flags)

AC_ARG_WITH([pyinstall_flags],
            [AS_HELP_STRING([--with-pyintall-flags],
                            [install flags for python (default='')])],
            [],
            [])
AC_SUBST(PYINSTALLFLAGS, $with_pyinstall_flags)

DX_DOT_FEATURE(OFF)
DX_HTML_FEATURE(ON)
DX_CHM_FEATURE(OFF)
DX_CHI_FEATURE(OFF)
DX_MAN_FEATURE(ON)
DX_RTF_FEATURE(OFF)
DX_XML_FEATURE(OFF)
DX_PDF_FEATURE(ON)
DX_PS_FEATURE(ON)
DX_INIT_DOXYGEN([bifrost])

AC_CONFIG_FILES([user2.mk])

AC_OUTPUT
