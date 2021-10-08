AC_INIT([bifrost], [], [], [], [https://github.com/ledatelescope/bifrost/])
AC_LANG(C++)
AC_CONFIG_SRCDIR([src/cuda.cpp])

AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([config])

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
       AC_MSG_ERROR([exhuberant ctags is required, but ${CTAGS} is a different version])],
      [AC_MSG_RESULT([yes])])

AC_SUBST(SO_EXT, $shrext_cmds)

#
# Features
#

AC_C_INLINE
AX_CXX_COMPILE_STDCXX(11, noext, mandatory)
AC_CHECK_FUNCS([memset])
AC_CHECK_FUNCS([rint])
AC_CHECK_FUNCS([socket])
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

AX_OPENMP
AS_IF([test x$HAVE_OPENMP != x1],
      [CPPFLAGS="$CPPFLAGS -DBF_OPENMP=0"
       CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"])

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

AC_MSG_CHECKING([if sys/socket.h supports recvmmsg])
AC_COMPILE_IFELSE([#include<sys/socket.h>
                   int main(void) {
                      struct mmsghdr *mmsg = NULL;
                      return 0;
                   }],
                   [AC_MSG_RESULT([yes])
                    AC_SUBST(HAVE_MMSG, 1)],
                   [AC_MSG_RESULT([no])])

#
# Extras
#

AC_ARG_WITH([cuda_home],
            [AS_HELP_STRING([--with-cuda-home],
                            [CUDA install path (default=/usr/local/cuda)])],
            [],
            [with_cuda_home=/usr/local/cuda])
AC_SUBST(CUDA_HOME, $with_cuda_home)
AC_ARG_ENABLE([cuda],
              [AS_HELP_STRING([--disable-cuda],
                              [disable cuda support (default=no)])],
              [enable_cuda=no],
              [enable_cuda=yes])
AS_IF([test x$enable_cuda != xno],
      [AX_WITH_PROG(NVCC, nvcc, [], [AC_SUBST(HAVE_CUDA, 0)])
       AX_WITH_PROG(NVPRUNE, nvprune, [], [AC_SUBST(HAVE_CUDA, 0)])
       AX_WITH_PROG(CUOBJDUMP, cuobjdump, [], [AC_SUBST(HAVE_CUDA, 0)])],
      [AC_SUBST(HAVE_CUDA, 0)])
AS_IF([test x$HAVE_CUDA = x1],
       [CPPFLAGS="$CPPFLAGS -DBF_CUDA_ENABLED=1"
        LDFLAGS="$LDFLAGS -L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
        LIBS="$LIBS -lcuda -lcudart -lnvrtc -lcublas -lcudadevrt -L. -lcufft_static_pruned -lculibos -lnvToolsExt"])
AC_ARG_WITH([nvcc_flags],
            [AS_HELP_STRING([--with-nvcc-flags],
                            [flags to pass to NVCC (default='-O3 -Xcompiler "-Wall"')])],
            [],
            [with_nvcc_flags='-O3 -Xcompiler "-Wall"'])
AC_SUBST(NVCCFLAGS, $with_nvcc_flags)


AC_ARG_ENABLE([numa],
              AS_HELP_STRING([--enable-numa],
                             [enable NUMA support (default=yes)]),
              [],
              [enable_numa=yes])
AS_IF([test x$enable_numa != xno],
      [AC_CHECK_LIB([numa], [numa_tonode_memory],
                    [CPPFLAGS="$CPPFLAGS -DBF_NUMA_ENABLED=1"
                     LIBS="$LIBS -lnuma"])])

AC_ARG_ENABLE([hwloc],
              AS_HELP_STRING([--enable-hwloc],
                             [enable hwloc support (default=yes)]),
              [],
              [enable_hwloc=yes])
AS_IF([test x$enable_hwloc != xno],
      [AC_CHECK_LIB([hwloc], [hwloc_topology_init],
                    [CPPFLAGS="$CPPFLAGS -DBF_HWLOC_ENABLED=1"
                     LIBS="$LIBS -lhwloc"])])

AC_ARG_ENABLE([vma],
              AS_HELP_STRING([--enable-vma],
                             [enable vma support (default=no)]),
              [],
              [enable_vma=no])
AS_IF([test x$enable_vma != xno],
      [AC_CHECK_LIB([vma], [recvfrom_zcopy],
                    [CPPFLAGS="$CPPFLAGS -DBF_VMA_ENABLED=1"
                     LIBS="$LIBS -lvma"])])

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
AS_IF([test x$HAVE_CUDA = x1],
      [NVCCFLAGS="$NVCCFLAGS -DBF_GPU_SHAREDMEM=$with_shared_mem"])

AC_ARG_WITH([alignment],
            [AS_HELP_STRING([--with-alignment=N],
                            [default memory alignment in bytes (default=4096)])],
            [],
            [with_alignment=4096])
CPPFLAGS="$CPPFLAGS -DBF_ALIGNMENT=$with_alignment"

AC_ARG_ENABLE([debug],
              [AS_HELP_STRING([--enable-debug],
                              [enable debugging mode (default=no)])],
              [enable_debug=yes],
              [enable_debug=no])
AS_IF([test x$enable_debug != xno],
      [CPPFLAGS="$CPPFLAGS -DBF_DEBUG=1"
       CXXFLAGS="$CXXFLAGS -g"
       NVCCFLAGS="$NVCCFLAGS -g"])

AC_ARG_ENABLE([trace],
              [AS_HELP_STRING([--enable-trace],
                              [enable tracing mode for nvprof/nvvp (default=no)])],
              [enable_trace=yes],
              [enable_trace=no])
AS_IF([test x$enable_trace != xno],
      [CPPFLAGS="$CPPFLAGS -DBF_TRACE_ENABLED=1"])

AC_ARG_ENABLE([native_arch],
              [AS_HELP_STRING([--disable-native-arch],
                              [disable native architecture compilation (default=no)])],
              [enable_native_arch=no],
              [enable_native_arch=yes])
AS_IF([test x$enable_native_arch != xyes],
      [],
      [CXXFLAGS="$CXXFLAGS -march=native"
       NVCCFLAGS="$NVCCFLAGS -Xcompiler \"-march=native\""])

AC_ARG_ENABLE([cuda_debug],
              [AS_HELP_STRING([--enable-cuda-debug],
                              [enable CUDA debugging (nvcc -G; default=no)])],
              [enable_cuda_debug=yes],
              [enable_cuda_debug=no])
AS_IF([test x$enable_cuda_debug != xno],
      [NVCCFLAGS="$NVCCFLAGS -G"])

#
# Python
#

AX_WITH_PROG(PYTHON, python)
AS_IF([test x${PYTHON} != x],
      [AC_MSG_CHECKING([whether $PYTHON as ctypesgen])
       AS_IF([! test $(PYTHON) -c "import ctypesgen"],
             [AC_MSG_RESULT([no])
              AC_MSG_WARN([python module will not be built])],
             [AC_MSG_RESULT([yes])
              AC_SUBST(HAVE_PYTHON, 1)])])

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

AX_WITH_PROG(DOCKER, docker)
AS_IF([test x${PYTHON} != x],
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

CXXFLAGS="$CXXFLAGS $lt_prog_compiler_pic_CXX"
NVCCFLAGS="$NVCCFLAGS -Xcompiler \"$lt_prog_compiler_pic_CXX\""

AC_CONFIG_FILES([config.mk Makefile src/Makefile python/Makefile])

AC_OUTPUT
