AC_DEFUN([AX_CHECK_CXX_ENDS_WITH],
[
  AC_PROVIDE([AX_CHECK_CXX_ENDS_WITH])
  
  AC_SUBST([HAVE_CXX_ENDS_WITH], [0])
  
  AC_MSG_CHECKING([for C++ std::string::ends_with support])
  
  AC_COMPILE_IFELSE([
    AC_LANG_PROGRAM([[
        #include <string>]],
        [[(std::string("This is a message")).ends_with("suffix")]])],
      [AC_SUBST([HAVE_CXX_ENDS_WITH], [1])],
      [AC_SUBST([HAVE_CXX_ENDS_WITH], [0])])
  
  if test "$HAVE_CXX_ENDS_WITH" = "1"; then
    AC_LINK_IFELSE([
      AC_LANG_PROGRAM([[
      #include <string>]],
      [[(std::string("This is a message")).ends_with("suffix")]])],
        [AC_SUBST([HAVE_CXX_ENDS_WITH], [1])],
        [AC_SUBST([HAVE_CXX_ENDS_WITH], [0])])
  fi
  
  if test "$HAVE_CXX_ENDS_WITH" = "1"; then
    AC_MSG_RESULT([yes])
    CXXFLAGS="-DHAVE_CXX_ENDS_WITH=1 $CXXFLAGS"
  else
    AC_MSG_RESULT([no])
  fi
])
