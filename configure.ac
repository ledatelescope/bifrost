AC_INIT([bifrost], [0.10.0], [], [], [https://github.com/ledatelescope/bifrost/])
AC_LANG(C++)
AC_CONFIG_SRCDIR([src/cuda.cpp])

AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([config])

m4_ifdef([AM_SILENT_RULES], [], [AC_SUBST([AM_DEFAULT_VERBOSITY], [1])])

: ${CXXFLAGS="-O3 -Wall -pedantic"}

#
# Programs
#

LT_INIT
AC_PROG_CC
AC_PROG_CXX
AC_PROG_AWK
AC_PROG_SED
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AX_WITH_PROG(CTAGS, ctags)
AS_IF([test x${CTAGS} = x],
      [AC_MSG_ERROR([Required program ctags was not found])],
      [])
AC_MSG_CHECKING([whether ${CTAGS} is exuberant])
AS_IF([! ${CTAGS} --version | grep -q Exuberant],
      [AC_MSG_RESULT([no])
       AC_MSG_ERROR([exuberant ctags is required, but ${CTAGS} is a different version])],
      [AC_MSG_RESULT([yes])])

AC_SUBST(SO_EXT, $shrext_cmds)

#
# System/Compiler Features
#

AC_C_INLINE
AX_CXX_COMPILE_STDCXX(20, noext, optional)
AS_IF([test x$HAVE_CXX20 != x1],
      [AX_CXX_COMPILE_STDCXX(17, noext, optional)
       AS_IF([test x$HAVE_CXX17 != x1],
             [AX_CXX_COMPILE_STDCXX(14, noext, optional)
              AS_IF([test x$HAVE_CXX14 != x1],
                    [AX_CXX_COMPILE_STDCXX(11, noext, mandatory)])])])
AX_CHECK_CXX_FILESYSTEM
AX_CHECK_CXX_ENDS_WITH
AC_CHECK_FUNCS([memset])
AC_CHECK_FUNCS([rint])
AC_CHECK_FUNCS([socket])
AC_CHECK_FUNCS([recvmsg],
               [AC_SUBST([HAVE_RECVMSG], [1])],
               [AC_SUBST([HAVE_RECVMSG], [0])])
AC_CHECK_FUNCS([sqrt])
AC_CHECK_FUNCS([strerror])
AC_CHECK_HEADERS([arpa/inet.h])
AC_CHECK_HEADERS([netdb.h])
AC_CHECK_HEADERS([netinet/in.h])
AC_CHECK_HEADERS([sys/file.h])
AC_CHECK_HEADERS([sys/ioctl.h])
AC_CHECK_HEADERS([sys/socket.h])
AC_CHECK_HEADER_STDBOOL
AC_FUNC_MALLOC

AC_SUBST(HAVE_OPENMP, 0)
AX_OPENMP
AS_IF([test x$OPENMP_CXXFLAGS != x],
      [AC_SUBST(HAVE_OPENMP, 1)])
AS_IF([test x$HAVE_OPENMP != x1],
      [],
      [CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"
       LDFLAGS="$LDFLAGS $OPENMP_CXXFLAGS"])

AC_CHECK_TYPES([ptrdiff_t])
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_INT8_T
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T
AC_TYPE_LONG_DOUBLE_WIDER
AC_SUBST([HAVE_FLOAT128], [0])
AS_IF([test x$HAVE_HAVE_LONG_DOUBLE_WIDER = x1],
      [AC_SUBST([HAVE_FLOAT128], [0])])

#
# NUMA
#

AC_ARG_ENABLE([numa],
              AS_HELP_STRING([--disable-numa],
                             [disable numa support (default=no)]),
              [enable_numa=no],
              [enable_numa=yes])
AC_SUBST([HAVE_NUMA], [0])
AS_IF([test x$enable_numa != xno],
      [AC_CHECK_LIB([numa], [numa_node_of_cpu],
                    [AC_SUBST([HAVE_NUMA], [1])
                     LIBS="$LIBS -lnuma"])])

#
# HWLOC
#

AC_ARG_ENABLE([hwloc],
              AS_HELP_STRING([--disable-hwloc],
                             [disable hwloc support (default=no)]),
              [enable_hwloc=no],
              [enable_hwloc=yes])
AC_SUBST([HAVE_HWLOC], [0])
AS_IF([test x$enable_hwloc != xno],
      [AC_CHECK_LIB([hwloc], [hwloc_topology_init],
                    [AC_SUBST([HAVE_HWLOC], [1])
                     LIBS="$LIBS -lhwloc"])])

#
# VMA
#

AC_ARG_ENABLE([vma],
              AS_HELP_STRING([--enable-vma],
                             [enable vma support (default=no)]),
              [enable_vma=yes],
              [enable_vma=no])
AC_SUBST([HAVE_VMA], [0])
AS_IF([test x$enable_vma != xno],
      [AC_CHECK_LIB([vma], [recvfrom_zcopy],
                    [AC_SUBST([HAVE_VMA], [1])
                     LIBS="$LIBS -lvma"])
       AC_CHECK_LIB([vma], [vma_recvfrom_zcopy],
                    [AC_SUBST([HAVE_VMA], [1])
                     LIBS="$LIBS -lvma"])])

#
# CUDA
#

#################################
#            NOTE               #
# This needs to come after all  #
# other compiler/library tests  #
# since it changes LIB to       #
# include CUDA-specific entries #
#################################

AX_CHECK_CUDA

AC_ARG_WITH([shared_mem],
           [AS_HELP_STRING([--with-shared-mem=N],
                           [default GPU shared memory in bytes (default=16384)])],
           [],
           [with_shared_mem=16384])
AC_SUBST([GPU_SHAREDMEM], [$with_shared_mem])
AS_IF([test x$HAVE_CUDA = x0],
      [AC_SUBST([GPU_SHAREDMEM], [0])])

#
# Bifrost memory alignment
#

AC_ARG_WITH([alignment],
            [AS_HELP_STRING([--with-alignment=N],
                            [default memory alignment in bytes (default=4096)])],
            [],
            [with_alignment=4096])
AC_SUBST([ALIGNMENT], [$with_alignment])

#
# Bifrost proclog location
#

AX_CHECK_TMPFS

#
# Bifrost Features
#

AC_ARG_ENABLE([debug],
              [AS_HELP_STRING([--enable-debug],
                              [enable debugging mode (default=no)])],
              [enable_debug=yes],
              [enable_debug=no])
AC_SUBST([HAVE_DEBUG], [0])
AS_IF([test x$enable_debug != xno],
      [AC_SUBST([HAVE_DEBUG], [1])
       CXXFLAGS="$CXXFLAGS -g"
       NVCCFLAGS="$NVCCFLAGS -g"])

AC_ARG_ENABLE([trace],
              [AS_HELP_STRING([--enable-trace],
                              [enable tracing mode for nvprof/nvvp (default=no)])],
              [enable_trace=yes],
              [enable_trace=no])
AC_SUBST([HAVE_TRACE], [0])
AS_IF([test x$enable_trace != xno],
      [AC_SUBST([HAVE_TRACE], [1])])

AX_CHECK_NATIVE_ARCH

AC_ARG_ENABLE([cuda_debug],
              [AS_HELP_STRING([--enable-cuda-debug],
                              [enable CUDA debugging (nvcc -G; default=no)])],
              [enable_cuda_debug=yes],
              [enable_cuda_debug=no])
AC_SUBST([HAVE_CUDA_DEBUG], [0])
AS_IF([test x$enable_cuda_debug != xno],
      [AC_SUBST([HAVE_CUDA_DEBUG], [1])
       NVCCFLAGS="$NVCCFLAGS -G"])

AC_ARG_ENABLE([map_cache],
              [AS_HELP_STRING([--disable-map-cache],
                              [disable caching bifrost.map kernels (default=no)])],
              [enable_map_cache=no],
              [enable_map_cache=yes])
AC_SUBST([HAVE_MAP_CACHE], [0])
AS_IF([test x$enable_map_cache != xno],
      [AC_SUBST([HAVE_MAP_CACHE], [$HAVE_CUDA])])

#
# Python
#

AC_ARG_ENABLE([python],
              [AS_HELP_STRING([--disable-python],
                              [disable building the Python bindings (default=no)])],
              [enable_python=no],
              [enable_python=yes])
AC_SUBST([HAVE_PYTHON], [0])
AS_IF([test x$enable_python != xno],
      [AX_WITH_PROG(PYTHON, python, no, $PATH)
       AS_IF([test x${PYTHON} = xno],
             [AC_PATH_PROG(PYTHON3, python3, no, $PATH)
              AS_IF([test x${PYTHON3} != xno],
                    [AC_SUBST([PYTHON], [$PYTHON3])])])
       AS_IF([test x${PYTHON} != xno],
             [AC_MSG_CHECKING([whether $PYTHON as ctypesgen])
              AS_IF([! ${PYTHON} -c "import ctypesgen" 2>/dev/null],
                    [AC_MSG_RESULT([no])
                     AC_MSG_WARN([python module will not be built])],
                    [AC_MSG_RESULT([yes])
                     AC_SUBST(HAVE_PYTHON, 1)])])])
AC_ARG_WITH([pybuild_flags],
            [AS_HELP_STRING([--with-pybuild-flags],
                            [build flags for python (default='')])],
            [],
            [])
AC_SUBST(PYBUILDFLAGS, $with_pybuild_flags)

AC_ARG_WITH([pyinstall_flags],
            [AS_HELP_STRING([--with-pyinstall-flags],
                            [install flags for python (default='')])],
            [],
            [])
AC_SUBST(PYINSTALLFLAGS, $with_pyinstall_flags)

#
# Docker
#

AX_WITH_PROG(DOCKER, docker, no, $PATH)
AS_IF([test x${DOCKER} != xno],
      [AC_SUBST(HAVE_DOCKER, 1)])

#
# Documentation
#

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

#
# Version splitting
#

AC_SUBST([PACKAGE_VERSION_MAJOR], [`echo $PACKAGE_VERSION | $AWK -F. '{print $1}'`])
AC_SUBST([PACKAGE_VERSION_MINOR], [`echo $PACKAGE_VERSION | $AWK -F. '{print $2}'`])
AC_SUBST([PACKAGE_VERSION_MICRO], [`echo $PACKAGE_VERSION | $AWK -F. '{print $3}'`])

#
# C++17/C++14/C++11 toggling
#

AC_SUBST([MAP_KERNEL_STDCXX], [c++11])
AC_SUBST([STDCXX_IS_SET], [0])
AS_IF([test x$HAVE_CXX20 = x1],
      [AC_SUBST([STDCXX_IS_SET], [1])
       CXXFLAGS="-std=c++20 $CXXFLAGS"
       AS_IF([test x$CUDA_HAVE_CXX20 = x1],
             [NVCCFLAGS="-std=c++20 $NVCCFLAGS"
              AC_SUBST([MAP_KERNEL_STDCXX], [c++20])],
             [AS_IF([test x$CUDA_HAVE_CXX17 = x1],
                    [NVCCFLAGS="-std=c++17 $NVCCFLAGS"
                     AC_SUBST([MAP_KERNEL_STDCXX], [c++17])],
                    [AS_IF([test x$CUDA_HAVE_CXX14 = x1],
                           [NVCCFLAGS="-std=c++14 $NVCCFLAGS"
                            AC_SUBST([MAP_KERNEL_STDCXX], [c++14])],
                           [NVCCFLAGS="-std=c++11 $NVCCFLAGS"])])])])
AS_IF([test x$HAVE_CXX17 = x1],
      [AC_SUBST([STDCXX_IS_SET], [1])
       CXXFLAGS="-std=c++17 $CXXFLAGS"
       AS_IF([test x$CUDA_HAVE_CXX17 = x1],
             [NVCCFLAGS="-std=c++17 $NVCCFLAGS"
              AC_SUBST([MAP_KERNEL_STDCXX], [c++17])],
             [AS_IF([test x$CUDA_HAVE_CXX14 = x1],
                    [NVCCFLAGS="-std=c++14 $NVCCFLAGS"
                     AC_SUBST([MAP_KERNEL_STDCXX], [c++14])],
                    [NVCCFLAGS="-std=c++11 $NVCCFLAGS"])])])
AS_IF([test x$HAVE_CXX14 = x1],
      [AC_SUBST([STDCXX_IS_SET], [1])
       CXXFLAGS="-std=c++14 $CXXFLAGS"
       AS_IF([test x$CUDA_HAVE_CXX14 = x1],
             [NVCCFLAGS="-std=c++14 $NVCCFLAGS"
              AC_SUBST([MAP_KERNEL_STDCXX], [c++14])],
             [NVCCFLAGS="-std=c++11 $NVCCFLAGS"])])
AS_IF([test x$STDCXX_IS_SET != x1],
      [AC_SUBST([STDCXX_IS_SET], [1])
       CXXFLAGS="-std=c++11 $CXXFLAGS"
       NVCCFLAGS="-std=c++11 -Xcompiler \"-DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT\" $NVCCFLAGS"])

#
# Linking flags
#

CXXFLAGS="$CXXFLAGS $lt_prog_compiler_pic_CXX"
NVCCFLAGS="$NVCCFLAGS -Xcompiler \"$lt_prog_compiler_pic_CXX\""

AC_CONFIG_FILES([config.mk Makefile src/Makefile python/Makefile share/bifrost.pc src/bifrost/config.h])

AC_OUTPUT

#
# User warnings
#

#
# User notes
#

echo ""

AS_IF([test x$HAVE_CUDA = x1],
      [AC_MSG_NOTICE(cuda: yes - v$CUDA_VERSION - $GPU_ARCHS - $with_stream_model streams)],
      [AC_MSG_NOTICE(cuda: no)])

AS_IF([test x$HAVE_NUMA = x1],
      [AC_MSG_NOTICE(numa: yes)],
      [AC_MSG_NOTICE(numa: no)])

AS_IF([test x$HAVE_HWLOC = x1],
      [AC_MSG_NOTICE(hwloc: yes)],
      [AC_MSG_NOTICE(hwloc: no)])

AS_IF([test x$HAVE_VMA = x1],
      [AC_MSG_NOTICE(libvma: yes)],
      [AC_MSG_NOTICE(libvma: no)])

AS_IF([test x$HAVE_PYTHON = x1],
      [AC_MSG_NOTICE(python bindings: yes)],
      [AC_MSG_NOTICE(python bindings: no)])

AC_MSG_NOTICE(memory alignment: $ALIGNMENT)
      
AC_MSG_NOTICE(logging directory: $HAVE_TMPFS)

AC_SUBST([OPTIONS], [])
AS_IF([test x$enable_debug != xno],
      [AC_SUBST([OPTIONS], ["$OPTIONS debug"])])
AS_IF([test x$enable_trace != xno],
      [AC_SUBST([OPTIONS], ["$OPTIONS trace"])])
AS_IF([test x$enable_cuda_debug != xno],
      [AC_SUBST([OPTIONS], ["$OPTIONS cuda_debug"])])
AS_IF([test x$enable_native_arch != xno],
      [AC_SUBST([OPTIONS], ["$OPTIONS native"])])
AS_IF([test x$HAVE_FLOAT128 != x0],
      [AC_SUBST([OPTIONS], ["$OPTIONS float128"])])
AS_IF([test x$enable_map_cache != xno],
      [AC_SUBST([OPTIONS], ["$OPTIONS map_cache"])])
AC_MSG_NOTICE(options:$OPTIONS)
      
echo ""
echo "Bifrost is now ready to be compiled.  Please run 'make'"
echo ""
