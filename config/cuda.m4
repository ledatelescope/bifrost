AC_DEFUN([AX_CHECK_CUDA],
[
  AC_PROVIDE([AX_CHECK_CUDA])
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
  
  AC_SUBST([HAVE_CUDA], [0])
  if test "$enable_cuda" != "no"; then
    AC_SUBST([HAVE_CUDA], [1])
    
    AC_PATH_PROG(NVCC, nvcc, no, [$CUDA_HOME/bin:$PATH])
    AC_PATH_PROG(NVPRUNE, nvprune, no, [$CUDA_HOME/bin:$PATH])
    AC_PATH_PROG(CUOBJDUMP, cuobjdump, no, [$CUDA_HOME/bin:$PATH])
  fi

  if test "$HAVE_CUDA" = "1"; then
    AC_MSG_CHECKING([for a working CUDA installation])
    
    CXXFLAGS_save="$CXXFLAGS"
    LDFLAGS_save="$LDFLAGS"
    LIBS_save="$LIBS"
    
    ac_compile='$NVCC -c $NVCCFLAGS conftest.$ac_ext >&5'
    AC_COMPILE_IFELSE([
      AC_LANG_PROGRAM([[
          #include <cuda.h>
          #include <cuda_runtime.h>]],
          [[cudaMalloc(0, 0);]])],
        [AC_MSG_RESULT(yes)],
        [AC_MSG_RESULT(no)
         AC_SUBST([HAVE_CUDA], [0])])
    
    CXXFLAGS="$CXXFLAGS_save"
    LDFLAGS="$LDFLAGS_save"
    LIBS="$LIBS_save"
  fi
  
  AC_ARG_WITH([nvcc_flags],
              [AS_HELP_STRING([--with-nvcc-flags],
                              [flags to pass to NVCC (default='-O3 -Xcompiler "-Wall"')])],
              [],
              [with_nvcc_flags='-O3 -Xcompiler "-Wall"'])
  AC_SUBST(NVCCFLAGS, $with_nvcc_flags)
  
  if test "$HAVE_CUDA" = "1"; then
    CPPFLAGS="$CPPFLAGS -DBF_CUDA_ENABLED=1"
    CXXFLAGS="$CXXFLAGS -DBF_CUDA_ENABLED=1"
    NVCCFLAGS="$NVCCFLAGS -DBF_CUDA_ENABLED=1"
    LDFLAGS="$LDFLAGS -L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
    LIBS="$LIBS -lcuda -lcudart -lnvrtc -lcublas -lcudadevrt -L. -lcufft_static_pruned -lculibos -lnvToolsExt"
  fi
  
])
