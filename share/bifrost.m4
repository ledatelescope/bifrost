AC_DEFUN([AX_CHECK_BIFROST],
[
  AC_PROVIDE([AX_CHECK_BIFROST])
  AC_ARG_WITH([bifrost],
              [AS_HELP_STRING([--with-bifrost],
                              [Bifrost install path (default=/usr/local)])],
              [],
              [with_bifrost=/usr/local/])
  AC_SUBST([BIFROST_PATH], [$with_bifrost])
  
  AC_SUBST([HAVE_BIFROST], [1])
  AC_CHECK_HEADER([bifrost/config.h], [], [AC_SUBST([HAVE_BIFROST], [0])])
  if test "$HAVE_BIFROST" = "1"; then
    AC_MSG_CHECKING([for a working Bifrost installation])
    
    CPPFLAGS_save="$CPPFLAGS"
    CXXFLAGS_save="$CXXFLAGS"
    LDFLAGS_save="$LDFLAGS"
    LIBS_save="$LIBS"
    
    CPPFLAGS="$CPPFLAGS -L$with_bifrost"
    LDFLAGS="$LDFLAGS -L$with_bifrost"
    LIBS="$LIBS -lbifrost"
    
    AC_COMPILE_IFELSE([
      AC_LANG_PROGRAM([[
          #include <bifrost/config.h>
          #include <bifrost/common.h>]],
          [[bfGetCudaEnabled();]])],
        [AC_MSG_RESULT(yes)],
        [AC_MSG_RESULT(no)
         AC_SUBST([HAVE_BIFROST], [0])])
    
    CPPFLAGS="$CPPFLAGS_save"
    CXXFLAGS="$CXXFLAGS_save"
    LDFLAGS="$LDFLAGS_save"
    LIBS="$LIBS_save"
  fi
  
  if test "$HAVE_BIFROST" = "1"; then
    CPPFLAGS="$CPPFLAGS -L$with_bifrost"
    LDFLAGS="$LDFLAGS -L$with_bifrost"
    LIBS="$LIBS -lbifrost"
  fi
])
