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
        [],
        [AC_SUBST([HAVE_CUDA], [0])])
    
    if test "$HAVE_CUDA" = "1"; then
      LDFLAGS="-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
      LIBS="$LIBS -lcuda -lcudart"

      ac_link='$NVCC -o conftest$ac_exeext $NVCCFLAGS $LDFLAGS $LIBS conftest.$ac_ext >&5'
      AC_LINK_IFELSE([
        AC_LANG_PROGRAM([[
            #include <cuda.h>
            #include <cuda_runtime.h>]],
            [[cudaMalloc(0, 0);]])],
          [AC_MSG_RESULT(yes)],
          [AC_MSG_RESULT(no)
           AC_SUBST([HAVE_CUDA], [0])])
    else
      AC_MSG_RESULT(no)
      AC_SUBST([HAVE_CUDA], [0])
    fi
    
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
  
  AC_ARG_WITH([gpu_archs],
              [AS_HELP_STRING([--with-gpu-archs=...],
                              [default GPU architectures (default=dectect)])],
              [],
              [with_gpu_archs='auto'])
  if test "$HAVE_CUDA" = "1"; then
    if test "$with_gpu_archs" = "auto"; then
      AC_MSG_CHECKING([which CUDA architectures to target])

      CXXFLAGS_save="$CXXFLAGS"
      LDFLAGS_save="$LDFLAGS"
      LIBS_save="$LIBS"
      
      LDFLAGS="-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
      LIBS="-lcuda -lcudart"
      ac_run='$NVCC -o conftest$ac_ext $LDFLAGS $LIBS conftest.$ac_ext>&5'
      AC_RUN_IFELSE([
        AC_LANG_PROGRAM([[
            #include <cuda.h>
            #include <cuda_runtime.h>
            #include <iostream>
            #include <fstream>
            #include <set>]],
            [[
            std::set<int> archs;
            int major, minor, arch;
            int deviceCount = 0;
            cudaGetDeviceCount(&deviceCount);
            if( deviceCount == 0 ) {
              return 1;
            }
            std::ofstream fh;
            fh.open("confarchs.out");
            for(int dev=0; dev<deviceCount; dev++) {
              cudaSetDevice(dev);
              cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
              cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
              arch = 10*major;
              if( archs.count(arch) == 0 ) {
                archs.insert(arch);
                if( dev > 0 ) {
                  fh << " ";
                }
                fh << arch;
              }
              arch += minor;
              if( archs.count(arch) == 0 ) {
                archs.insert(arch);
                if( dev > 0 ) {
                  fh << " ";
                }
                fh << arch;
              }
            }
            fh.close();]])],
            [AC_SUBST([GPU_ARCHS], [`cat confarchs.out`])
             AC_MSG_RESULT([$GPU_ARCHS])],
            [AC_MSG_ERROR(failed to find any)])

      CXXFLAGS="$CXXFLAGS_save"
      LDFLAGS="$LDFLAGS_save"
      LIBS="$LIBS_save"
    else
      AC_SUBST([GPU_ARCHS], [$with_gpu_archs])
    fi
  fi 
])
