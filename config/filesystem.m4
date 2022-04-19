AC_DEFUN([AX_CHECK_CXX_FILESYSTEM],
[
  AC_PROVIDE([AX_CHECK_CXX_FILESYSTEM])
  
  AC_SUBST([HAVE_CXX_FILESYSTEM], [0])
  
  AC_MSG_CHECKING([for C++ std::filesystem support])
  
  AC_COMPILE_IFELSE([
    AC_LANG_PROGRAM([[
        #include <filesystem>]],
        [[std::filesystem::exists("/")]])],
      [AC_SUBST([HAVE_CXX_FILESYSTEM], [1])],
      [AC_SUBST([HAVE_CXX_FILESYSTEM], [0])])
  
  if test "$HAVE_CXX_FILESYSTEM" = "1"; then
    AC_LINK_IFELSE([
      AC_LANG_PROGRAM([[
          #include <filesystem>]],
          [[std::filesystem::exists("/")]])],
        [AC_SUBST([HAVE_CXX_FILESYSTEM], [1])],
        [AC_SUBST([HAVE_CXX_FILESYSTEM], [0])])
  fi
  
  if test "$HAVE_CXX_FILESYSTEM" = "1"; then
    AC_MSG_RESULT([yes])
    CXXFLAGS="-DHAVE_CXX_FILESYSTEM=1 $CXXFLAGS"
  else
    AC_MSG_RESULT([no])
  fi
])
