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

    AC_SUBST([GPU_SHAREDMEM], 0)
    AC_ARG_WITH([shared_mem],
        [AS_HELP_STRING([--with-shared-mem=N],
                        [default GPU shared memory per block in bytes (default=detect)])],
        [AC_SUBST([GPU_SHAREDMEM], [$withval])],
        [with_shared_mem='auto'])
    AC_MSG_NOTICE([-with-shared-mem=$GPU_SHAREDMEM])

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

        AS_IF([test "x$with_shared_mem" = "xauto"], [
            AC_MSG_CHECKING([GPU shared memory using automatic method])
            ac_compile='$HIPCC -c $HIPCCFLAGS conftest.$ac_ext >&5'

            AC_COMPILE_IFELSE([
                AC_LANG_PROGRAM([[
                    #include <algorithm>
                    #include <fstream>
                    #include <iostream>
                    #include <limits>
                    #include <hip/hip_runtime.h>
                ]], [[
                    int count {};
                    auto hiperr = hipGetDeviceCount(&count);
                    if (hiperr != hipSuccess) {
                        std::cerr << "Error detecting devices" << std::endl;
                        return 1;
                    }
                    if (count == 0) {
                        std::cerr << "No devices detected" << std::endl;
                        return 1;
                    }

                    size_t mem {std::numeric_limits<size_t>::max()};
                    for (int device = 0; device < count; ++device) {
                        hipDeviceProp_t prop;
                        hiperr = hipGetDeviceProperties(&prop, device);
                        if (hiperr != hipSuccess) {
                            std::cerr << "Failed to query shared memory for device " << device << std::endl;
                            return 1;
                        }
                        mem = std::min(mem, prop.sharedMemPerBlock);
                    }

                    std::ofstream fd;
                    fd.open("confmem.out");
                    fd << mem;
                    fd.close();

                    return 0;
                ]])
            ], [
                AC_SUBST([GPU_SHAREDMEM], [$(cat confmem.out)])
                AC_MSG_RESULT([$GPU_SHAREDMEM bytes])
            ], [
                AC_MSG_ERROR([failed])
            ])
        ])
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

    AC_MSG_NOTICE([hip config complete])
])