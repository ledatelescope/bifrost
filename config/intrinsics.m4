#
# SSE
#

AC_DEFUN([AX_CHECK_SSE],
[
  AC_PROVIDE([AX_CHECK_SSE])
  AC_ARG_ENABLE([sse],
                [AS_HELP_STRING([--disable-sse],
                                [disable SSE support (default=no)])],
                [enable_sse=no],
                [enable_sse=yes])

  AC_SUBST([HAVE_SSE], [0])
  
  if test "$enable_sse" = "yes"; then
    AC_MSG_CHECKING([for SSE support via '-msse'])
    
    CXXFLAGS_temp="$CXXFLAGS -msse"
    
    ac_run="$CXX -o conftest$ac_ext $CXXFLAGS_temp conftest.$ac_ext>&5"
    AC_RUN_IFELSE([
      AC_LANG_PROGRAM([[
        #include <xmmintrin.h>]],
        [[
        __m128 x = _mm_set1_ps(1.0f);
        x = _mm_add_ps(x, x);
        return _mm_cvtss_f32(x) != 2.0f;]])], 
    [
     CXXFLAGS="$CXXFLAGS -msse"
     AC_SUBST([HAVE_SSE], [1])
     AC_MSG_RESULT([yes])
    ], [
     AC_MSG_RESULT([no])
    ])
  fi
])



#
# AVX
#

AC_DEFUN([AX_CHECK_AVX],
[
  AC_PROVIDE([AX_CHECK_AVX])
  AC_ARG_ENABLE([avx],
                [AS_HELP_STRING([--disable-avx],
                                [disable AVX support (default=no)])],
                [enable_avx=no],
                [enable_avx=yes])
  
  AC_SUBST([HAVE_AVX], [0])
  
  if test "$enable_avx" = "yes"; then
    AC_MSG_CHECKING([for AVX support via '-mavx'])
    
    CXXFLAGS_temp="$CXXFLAGS -mavx"
    ac_run_save="$ac_run"
    
    ac_run="$CXX -o conftest$ac_ext $CXXFLAGS_temp conftest.$ac_ext>&5"
    AC_RUN_IFELSE([
      AC_LANG_PROGRAM([[
        #include <immintrin.h>]],
        [[
        __m256d x = _mm256_set1_pd(1.0);
        x = _mm256_add_pd(x, x);
        return _mm256_cvtsd_f64(x) != 2.0;]])],
    [
     CXXFLAGS="$CXXFLAGS -mavx"
     AC_SUBST([HAVE_AVX], [1])
     AC_MSG_RESULT([yes])
    ], [
     AC_MSG_RESULT([no])
    ])
    
    ac_run="$ac_run_save"
  fi
])

#
# AVX512
#

AC_DEFUN([AX_CHECK_AVX512],
[
  AC_PROVIDE([AX_CHECK_AVX512])
  AC_ARG_ENABLE([avx512],
                [AS_HELP_STRING([--disable-avx512],
                                [disable AVX512 support (default=no)])],
                [enable_avx512=no],
                [enable_avx512=yes])
  
  AC_SUBST([HAVE_AVX512], [0])
  
  if test "$enable_avx512" = "yes"; then
    AC_MSG_CHECKING([for AVX-512 support via '-mavx512f'])
    
    CXXFLAGS_temp="$CXXFLAGS -mavx512f"
    ac_run_save="$ac_run"
    
    ac_run="$CXX -o conftest$ac_ext $CXXFLAGS_temp conftest.$ac_ext>&5"
    AC_RUN_IFELSE([
      AC_LANG_PROGRAM([[
        #include <immintrin.h>]],
        [[
        __m512d x = _mm512_set1_pd(1.0);
        x = _mm512_add_pd(x, x);
        return _mm512_cvtsd_f64(x) != 2.0;]])], 
    [
     CXXFLAGS="$CXXFLAGS -mavx512f"
     AC_SUBST([HAVE_AVX512], [1])
     AC_MSG_RESULT([yes])
    ], [
     AC_MSG_RESULT([no])
    ])
    
    ac_run="$ac_run_save"
  fi
])
