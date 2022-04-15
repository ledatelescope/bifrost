AC_DEFUN([AX_CHECK_CXX_FILESYSTEM],
[
  AC_PROVIDE([AX_CHECK_CXX_FILESYSTEM])
  
  AC_SUBST([HAVE_CXX_FILESYSTEM], [0])
  
  AC_MSG_CHECKING([for a C++ std::filesystem support])
  
  CXXFLAGS_save="$CXXFLAGS"
  LDFLAGS_save="$LDFLAGS"
  LIBS_save="$LIBS"
  
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
  else
    AC_MSG_RESULT([no])
  fi
])
