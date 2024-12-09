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
AC_MSG_CHECKING([whether ${CTAGS} is sufficiently exuberant])
AS_IF([! ${CTAGS} --version | grep -q Exuberant],
      [AC_MSG_RESULT([no])
       AC_MSG_ERROR([exuberant/universal ctags is required, but ${CTAGS} is a different version])],
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
# Infiniband Verbs
#

AC_ARG_ENABLE([verbs],
              AS_HELP_STRING([--disable-verbs],
                             [disable Infiniband verbs support (default=no)]),
              [enable_verbs=no],
              [enable_verbs=yes])
AC_SUBST([HAVE_VERBS], [0])
AS_IF([test x$enable_verbs != xno],
      [AC_CHECK_LIB([ibverbs], [ibv_query_device],
                    [AC_SUBST([HAVE_VERBS], [1])
                     LIBS="$LIBS -libverbs"])])

 AC_ARG_WITH([rx-buffer-size],
            [AS_HELP_STRING([--with-rx-buffer-size=N],
                            [default Infiniband verbs receive buffer size in packets (default=8192)])],
            [],
            [with_rx_buffer_size=8192])
 AC_SUBST([VERBS_NPKTBUF], [$with_rx_buffer_size])
 AS_IF([test x$HAVE_VERBS = x0],
       [AC_SUBST([VERBS_NPKTBUF], [0])])

 AC_ARG_WITH([tx-buffer-size],
            [AS_HELP_STRING([--with-tx-buffer-size=N],
                            [default Infiniband verbs send buffer size in packets (default=512)])],
            [],
            [with_tx_buffer_size=512])
 AC_SUBST([VERBS_SEND_NPKTBUF], [$with_tx_buffer_size])
 AS_IF([test x$HAVE_VERBS = x0],
       [AC_SUBST([VERBS_SEND_NPKTBUF], [0])])

 AS_IF([test x$HAVE_VERBS != x0],
       [AC_MSG_CHECKING([for Infiniband verbs packet pacing support])
        AC_COMPILE_IFELSE([
          AC_LANG_PROGRAM([[
            #include <infiniband/verbs.h>]],
          [[
            int ndev, d;
            ibv_device** ibv_dev_list = NULL;
            ibv_context* ibv_ctx = NULL;
            ibv_device_attr_ex ibv_dev_attr;
            
            ibv_dev_list = ibv_get_device_list(&ndev);
            for(d=0; d<ndev; d++) {
                ibv_ctx = ibv_open_device(ibv_dev_list[d]);
                ibv_query_device_ex(ibv_ctx, NULL, &ibv_dev_attr);
                ibv_is_qpt_supported(ibv_dev_attr.packet_pacing_caps.supported_qpts, IBV_QPT_RAW_PACKET);
            }
          ]])],
          [AC_SUBST([VERBS_SEND_PACING], [1])
           AC_MSG_RESULT([yes])],
          [AC_SUBST([VERBS_SEND_PACING], [0])
           AC_MSG_RESULT([no])])
       ])

#
# RDMA
#

AC_ARG_ENABLE([rdma],
              AS_HELP_STRING([--disable-rdma],
                             [disable RDMA support (default=no)]),
              [enable_rdma=no],
              [enable_rdma=yes])
AC_SUBST([HAVE_RDMA], [0])
AS_IF([test x$enable_rdma != xno],
      [AC_CHECK_LIB([rdmacm], [rdma_get_devices],
                    [AC_SUBST([HAVE_RDMA], [1])
                     LIBS="$LIBS -lrdmacm"])])

AC_ARG_WITH([rdma_max_mem],
            AS_HELP_STRING([--with-rdma-max-mem=N],
                           [maximum RDMA buffer size in bytes (default=134217728)]),
            [],
            [with_rdma_max_mem=134217728])
AC_SUBST([RDMA_MAXMEM], [$with_rdma_max_mem])
AS_IF([test x$HAVE_RDMA = x0],
      [AC_SUBST([RDMA_MAXMEM], [0])])

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

AX_CHECK_SSE
AX_CHECK_AVX
AX_CHECK_AVX512
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
             [AC_MSG_CHECKING([if $PYTHON is version 3.6 or later])
              AS_IF([! ${PYTHON} -c "import sys; assert(sys.version_info >= (3,6,0))" 2>/dev/null],
                    [AC_MSG_RESULT([no])
                     AC_MSG_WARN([python module will not be built])
                     AC_SUBST([PYTHON], [no])],
                    [AC_MSG_RESULT([yes])])])
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

DX_INIT_DOXYGEN([bifrost])
DX_DOT_FEATURE(OFF)
DX_HTML_FEATURE(ON)
DX_CHM_FEATURE(OFF)
DX_CHI_FEATURE(OFF)
DX_MAN_FEATURE(ON)
DX_RTF_FEATURE(OFF)
DX_XML_FEATURE(OFF)
DX_PDF_FEATURE(ON)
DX_PS_FEATURE(ON)

AS_IF([test x${HAVE_PYTHON} == x1],
      [AX_WITH_PROG(PYTHON_SPHINXB, sphinx-build)
       AX_WITH_PROG(PYTHON_SPHINXA, sphinx-apidoc)
       AX_WITH_PROG(PYTHON_BREATHE, breathe-apidoc)])

AC_SUBST(HAVE_CXX_DOCS, 0)
AC_SUBST(HAVE_PYTHON_DOCS, 0)
AS_IF([test x${DX_DOXYGEN} == x],
      [AC_MSG_WARN([missing doxygen, documentation cannot be built])],
      [AC_SUBST(HAVE_CXX_DOCS, 1)
       AS_IF([test x${PYTHON} != xno],
             [AS_IF([test x${PYTHON_SPHINXB} = x],
                    [AC_MSG_WARN([missing the sphinx-build, python documentation cannot not be built])],
                    [AS_IF([test x${PYTHON_SPHINXA} = x],
                            [AC_MSG_WARN([missing the sphinx-apidoc, python documentation cannot not be built])],
                            [AS_IF([test x${PYTHON_BREATHE} = x],
                                   [AC_MSG_WARN([missing the breathe-apidoc, python documentation cannot not be built])],
                                   [AC_SUBST(HAVE_PYTHON_DOCS, 1)])])])])])


# Version splitting
#

AC_SUBST([PACKAGE_VERSION_MAJOR], [`echo $PACKAGE_VERSION | $AWK -F. '{print $1}'`])
AC_SUBST([PACKAGE_VERSION_MINOR], [`echo $PACKAGE_VERSION | $AWK -F. '{print $2}'`])
AC_SUBST([PACKAGE_VERSION_MICRO], [`echo $PACKAGE_VERSION | $AWK -F. '{print $3}'`])

#
# C++20/C++17/C++14/C++11 toggling
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
       AS_IF([test x$CUDA_HAVE_CXX20 = x1],
             [NVCCFLAGS="-std=c++17 $NVCCFLAGS"
              AC_SUBST([MAP_KERNEL_STDCXX], [c++17])],
             [AS_IF([test x$CUDA_HAVE_CXX17 = x1],
                    [NVCCFLAGS="-std=c++17 $NVCCFLAGS"
                     AC_SUBST([MAP_KERNEL_STDCXX], [c++17])],
                    [AS_IF([test x$CUDA_HAVE_CXX14 = x1],
                           [NVCCFLAGS="-std=c++14 $NVCCFLAGS"
                            AC_SUBST([MAP_KERNEL_STDCXX], [c++14])],
                           [NVCCFLAGS="-std=c++11 $NVCCFLAGS"])])])])
AS_IF([test x$HAVE_CXX14 = x1],
      [AC_SUBST([STDCXX_IS_SET], [1])
       CXXFLAGS="-std=c++14 $CXXFLAGS"
       AS_IF([test x$CUDA_HAVE_CXX20 = x1],
             [NVCCFLAGS="-std=c++14 $NVCCFLAGS"
              AC_SUBST([MAP_KERNEL_STDCXX], [c++14])],
             [AS_IF([test x$CUDA_HAVE_CXX17 = x1],
                    [NVCCFLAGS="-std=c++14 $NVCCFLAGS"
                     AC_SUBST([MAP_KERNEL_STDCXX], [c++14])],
                    [AS_IF([test x$CUDA_HAVE_CXX14 = x1],
                           [NVCCFLAGS="-std=c++14 $NVCCFLAGS"
                            AC_SUBST([MAP_KERNEL_STDCXX], [c++14])],
                           [NVCCFLAGS="-std=c++11 $NVCCFLAGS"])])])])
AS_IF([test x$STDCXX_IS_SET != x1],
      [AC_SUBST([STDCXX_IS_SET], [1])
       CXXFLAGS="-std=c++11 $CXXFLAGS"
       NVCCFLAGS="-std=c++11 -Xcompiler \"-DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT\" $NVCCFLAGS"])

#
# Linking flags
#

CXXFLAGS="$CXXFLAGS $lt_prog_compiler_pic_CXX"
NVCCFLAGS="$NVCCFLAGS -Xcompiler \"$lt_prog_compiler_pic_CXX\""
LIBS="$LIBS $NVCCLIBS"

#
# Additional CUDA flags
#
AC_SUBST([NVCC_GENCODE], [])
AS_IF([test x$HAVE_CUDA != x0],
      [NVCC_GENCODE=$(echo $GPU_ARCHS | ${SED} -e 's/\([[0-9]]\{2,3\}\)/-gencode arch=compute_\1,\\"code=sm_\1\\"/g;')
       NVCC_GENCODE="$NVCC_GENCODE -gencode arch=compute_${GPU_MAX_ARCH},\\\"code=compute_${GPU_MAX_ARCH}\\\""
       NVCCFLAGS="$NVCCFLAGS ${NVCC_GENCODE}"
       CPPFLAGS="$CPPFLAGS -I$CUDA_HOME/include"
      ])

AC_CONFIG_FILES([config.mk Makefile src/Makefile python/Makefile docs/Makefile share/bifrost.pc src/bifrost/config.h])

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

AS_IF([test x$HAVE_HWLOC = x1],
      [AC_MSG_NOTICE(hwloc: yes)],
      [AC_MSG_NOTICE(hwloc: no)])

AS_IF([test x$HAVE_VMA = x1],
      [AC_MSG_NOTICE(libvma: yes)],
      [AC_MSG_NOTICE(libvma: no)])

AS_IF([test x$HAVE_VERBS = x1],
      [AC_MSG_NOTICE(libverbs: yes)],
      [AC_MSG_NOTICE(libverbs: no)])

AS_IF([test x$HAVE_RDMA = x1],
      [AC_MSG_NOTICE(librdmacm: yes)],
      [AC_MSG_NOTICE(librdmacm: no)])
      
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
AS_IF([test x$HAVE_SSE != x0],
      [AC_SUBST([OPTIONS], ["$OPTIONS sse"])])
AS_IF([test x$HAVE_AVX != x0],
      [AC_SUBST([OPTIONS], ["$OPTIONS avx"])])
AS_IF([test x$HAVE_AVX512 != x0],
      [AC_SUBST([OPTIONS], ["$OPTIONS avx512"])])
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
