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
  AC_SUBST([CUDA_VERSION], [0])
  AC_SUBST([CUDA_HAVE_CXX20], [0])
  AC_SUBST([CUDA_HAVE_CXX17], [0])
  AC_SUBST([CUDA_HAVE_CXX14], [0])
  AC_SUBST([CUDA_HAVE_CXX11], [0])
  AC_SUBST([GPU_MIN_ARCH], [0])
  AC_SUBST([GPU_MAX_ARCH], [0])
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
          [CUDA_VERSION=$( ${NVCC} --version | ${GREP} -Po -e "release.*," | cut -d,  -f1 | cut -d\  -f2 )
           AC_MSG_RESULT(yes - v$CUDA_VERSION)],
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
  
  if test "$HAVE_CUDA" = "1"; then
    AC_MSG_CHECKING([for CUDA CXX standard support])
    
    CUDA_STDCXX=$( ${NVCC} --help | ${GREP} -Po -e "--std.*}" | ${SED} 's/.*|//;s/}//;' )
    if test "$CUDA_STDCXX" = "c++20"; then
      AC_MSG_RESULT(C++20)
      AC_SUBST([CUDA_HAVE_CXX20], [1])
    else
      if test "$CUDA_STDCXX" = "c++17"; then
        AC_MSG_RESULT(C++17)
        AC_SUBST([CUDA_HAVE_CXX17], [1])
      else
        if test "$CUDA_STDCXX" = "c++14"; then
          AC_MSG_RESULT(C++14)
          AC_SUBST([CUDA_HAVE_CXX14], [1])
        else
          if test "$CUDA_STDCXX" = "c++11"; then
            AC_MSG_RESULT(C++11)
            AC_SUBST([CUDA_HAVE_CXX11], [1])
          else
            AC_MSG_ERROR(nvcc does not support at least C++11)
          fi
        fi
      fi
    fi
  fi
  
  AC_ARG_WITH([nvcc_flags],
              [AS_HELP_STRING([--with-nvcc-flags],
                              [flags to pass to NVCC (default='-O3 -Xcompiler "-Wall"')])],
              [],
              [with_nvcc_flags='-O3 -Xcompiler "-Wall"'])
  AC_SUBST(NVCCFLAGS, $with_nvcc_flags)
  
  AC_ARG_WITH([stream_model],
              [AS_HELP_STRING([--with-stream-model],
                              [CUDA default stream model to use: 'legacy' or 'per-thread' (default='per-thread')])],
              [],
              [with_stream_model='per-thread'])
  
  
  if test "$HAVE_CUDA" = "1"; then
    AC_MSG_CHECKING([for different CUDA default stream models])
    dsm_supported=$( ${NVCC} -h | ${GREP} -Po -e "--default-stream" )
    if test "$dsm_supported" = "--default-stream"; then
      if test "$with_stream_model" = "per-thread"; then
        NVCCFLAGS="$NVCCFLAGS -default-stream per-thread"
        AC_MSG_RESULT([yes, using 'per-thread'])
      else
        if test "$with_stream_model" = "legacy"; then
          NVCCFLAGS="$NVCCFLAGS -default-stream legacy"
          AC_MSG_RESULT([yes, using 'legacy'])
        else
          AC_MSG_ERROR(Invalid CUDA stream model: '$with_stream_model')
        fi
      fi
    else
      AC_MSG_RESULT([no, only the 'legacy' stream model is supported])
    fi
  fi
  
  if test "$HAVE_CUDA" = "1"; then
    CPPFLAGS="$CPPFLAGS -DBF_CUDA_ENABLED=1"
    CXXFLAGS="$CXXFLAGS -DBF_CUDA_ENABLED=1"
    NVCCFLAGS="$NVCCFLAGS -DBF_CUDA_ENABLED=1"
    LDFLAGS="$LDFLAGS -L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
    LIBS="$LIBS -lcuda -lcudart -lnvrtc -lcublas -lcudadevrt -L. -lcufft_static_pruned -lculibos -lnvToolsExt"
  fi
  
  AC_ARG_WITH([gpu_archs],
              [AS_HELP_STRING([--with-gpu-archs=...],
                              [default GPU architectures (default=detect)])],
              [],
              [with_gpu_archs='auto'])
  if test "$HAVE_CUDA" = "1"; then
    AC_MSG_CHECKING([for valid CUDA architectures])
    ar_supported=$( ${NVCC} -h | ${GREP} -Po "'compute_[[0-9]]{2,3}" | cut -d_ -f2 | sort | uniq )
    ar_supported_flat=$( echo $ar_supported | xargs )
    AC_MSG_RESULT(found: $ar_supported_flat)
    
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
                fh << " " << arch;
              }
            }
            fh.close();]])],
            [AC_SUBST([GPU_ARCHS], [`cat confarchs.out`])
             ar_supported=$( ${NVCC} -h | ${GREP} -Po "'compute_[[0-9]]{2,3}" | cut -d_ -f2 | sort | uniq )
             ar_valid=$( echo $GPU_ARCHS $ar_supported | xargs -n1 | sort | uniq -d | xargs )
             if test "$ar_valid" = ""; then
               AC_MSG_ERROR(failed to find any supported)
             else
               AC_SUBST([GPU_ARCHS], [$ar_valid])
               AC_MSG_RESULT([$GPU_ARCHS])
             fi],
            [AC_MSG_ERROR(failed to find any)])

      CXXFLAGS="$CXXFLAGS_save"
      LDFLAGS="$LDFLAGS_save"
      LIBS="$LIBS_save"
    else
      AC_SUBST([GPU_ARCHS], [$with_gpu_archs])
    fi
    
    AC_MSG_CHECKING([for valid requested CUDA architectures])
    ar_requested=$( echo "$GPU_ARCHS" | wc -w )
    ar_valid=$( echo $GPU_ARCHS $ar_supported | xargs -n1 | sort | uniq -d | xargs )
    ar_found=$( echo $ar_valid | wc -w )
    if test "$ar_requested" = "$ar_found"; then
      AC_MSG_RESULT([yes])
    else
      AC_MSG_ERROR(only '$ar_valid' are supported)
    fi
    
    ar_min_valid=$(echo $ar_valid | ${SED} -e 's/ .*//g;' )
    AC_SUBST([GPU_MIN_ARCH], [$ar_min_valid])
    ar_max_valid=$(echo $ar_valid | ${SED} -e 's/.* //g;' )
    AC_SUBST([GPU_MAX_ARCH], [$ar_max_valid])

    AC_MSG_CHECKING([for Pascal-style CUDA managed memory])
    cm_invalid=$( echo $GPU_ARCHS | ${SED} -e 's/\b[[1-5]][[0-9]]\b/PRE/g;' )
    if ! echo $cm_invalid | ${GREP} -q PRE; then
      AC_SUBST([GPU_PASCAL_MANAGEDMEM], [1])
      AC_MSG_RESULT([yes])
    else
      AC_SUBST([GPU_PASCAL_MANAGEDMEM], [0])
      AC_MSG_RESULT([no])
    fi
  else
     AC_SUBST([GPU_PASCAL_MANAGEDMEM], [0])
  fi
])
