AC_DEFUN([AX_CHECK_HIP],
[
    AC_PROVIDE([AX_CHECK_HIP])

    AC_ARG_ENABLE(
        [gpu],
        [AS_HELP_STRING(
            [--enable-gpu],
            [enable GPU support (default=no)])
        ],
        [AC_SUBST([ENABLE_GPU], [1])],
        [AC_SUBST([ENABLE_GPU], [0])]
    )

    AC_MSG_CHECKING([if gpu is enabled])
    AC_MSG_RESULT([$ENABLE_GPU])

    AS_IF([test "x$ENABLE_GPU" = "x1"], [
        AC_PATH_PROG(HIPCONFIG, hipconfig, no)
        AS_IF([test "x$HIPCONFIG" = "xno"], [
            AC_MSG_ERROR("could not find hipconfig in path")
        ])

        AC_PATH_PROG(HIPCC, hipcc, no)
        AS_IF([test "x$HIPCC" = "xno"], [
            AC_MSG_ERROR("could not find hipcc in path")
        ])

        AC_MSG_CHECKING([for GPU platform])
        AC_SUBST([GPU_PLATFORM], [`hipconfig -P`])
        AC_MSG_RESULT([$GPU_PLATFORM])

        AC_MSG_CHECKING([for hip path])
        AC_SUBST([HIP_PATH], [`hipconfig -p`])
        AC_MSG_RESULT([$HIP_PATH])

        AC_MSG_CHECKING([for rocm path])
        AC_SUBST([ROCM_PATH], [`hipconfig -R`])
        AC_MSG_RESULT([$ROCM_PATH])

        AC_MSG_CHECKING([for hipcc C++ config])
        AC_SUBST([HIP_CPPCONF], [`hipconfig -C`])
        AC_MSG_RESULT([$HIP_CPPCONF])
    ])

    AC_SUBST([HIPCCFLAGS])

    AS_IF([test "x$GPU_PLATFORM" = "xnvidia"], [
        CXXFLAGS="$CXXFLAGS -DBF_CUDA_ENABLED=1 $HIP_CPPCONF"
        HIPCCFLAGS="$HIPCCFLAGS --std=c++17 -DBF_CUDA_ENABLED=1 $HIP_CPPCONF"
        LDFLAGS="$LDFLAGS -L$HIP_PATH/lib -L$ROCM_PATH/lib"
        LIBS="$LIBS -lhipfft -lhipblas"
    ])

    AS_IF([test "x$GPU_PLATFORM" = "xamd"], [
        CXXFLAGS="$CXXFLAGS -DBF_CUDA_ENABLED=1 $HIP_CPPCONF "
        HIPCCFLAGS="$HIPCCFLAGS -std=c++17 -Wall -O3 -DBF_CUDA_ENABLED=1 $HIP_CPPCONF"
        LDFLAGS="$LDFLAGS -L$HIP_PATH/lib -L$ROCM_PATH/lib"
        LIBS="$LIBS -lhipfft -lhipblas -lhiprtc"

        # AMD Constants
        AC_SUBST([GPU_MANAGEDMEM], [1])
        AC_SUBST([GPU_MIN_ARCH], [0])
        AC_SUBST([GPU_MAX_ARCH], [0])
        AC_SUBST([GPU_EXP_PINNED_ALLOC], [0])
        AC_SUBST([CUDA_VERSION], [0])
    ])
])