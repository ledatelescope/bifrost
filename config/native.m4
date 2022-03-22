AC_DEFUN([AX_CHECK_NATIVE_ARCH],
[
  AC_PROVIDE([AX_CHECK_NATIVE_ARCH])
  AC_ARG_ENABLE([native_arch],
                [AS_HELP_STRING([--disable-native-arch],
                                [disable native architecture compilation (default=no)])],
                [enable_native_arch=no],
                [enable_native_arch=yes])
  
  if test "$enable_native_arch" = "yes"; then
    AC_MSG_CHECKING([if the compiler accepts '-march=native'])
    
    CXXFLAGS_temp="$CXXFLAGS -march=native"
    
    ac_compile='$CXX -c $CXXFLAGS_temp conftest.$ac_ext >&5'
    AC_COMPILE_IFELSE([
      AC_LANG_PROGRAM([[
          ]],
          [[
          int i = 5;]])],
        [CXXFLAGS="$CXXFLAGS -march=native"
         NVCCFLAGS="$NVCCFLAGS -Xcompiler \"-march=native\""
         AC_MSG_RESULT(yes)],
        [AC_SUBST([enable_native_arch], [no])
         AC_MSG_RESULT(no)])
  fi
])
