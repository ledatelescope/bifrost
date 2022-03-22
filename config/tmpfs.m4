AC_DEFUN([AX_CHECK_TMPFS],
[
  AC_PROVIDE([AX_CHECK_TMPFS])
  
  AC_ARG_WITH([logging_dir],
              [AS_HELP_STRING([--with-logging-dir=[DIR]],
                              [directory for Bifrost proclog logging (default=autodetect)])],
              [AC_SUBST([HAVE_TMPFS], [$with_logging_dir])],
              [AC_SUBST([HAVE_TMPFS], [/tmp])])
  
  if test "$HAVE_TMPFS" = "/tmp"; then
    AC_CHECK_FILE([/dev/shm],
                  [AC_SUBST([HAVE_TMPFS], [/dev/shm/bifrost])])
  fi
  
  if test "$HAVE_TMPFS" = "/tmp"; then
    AC_CHECK_FILE([/Volumes/RAMDisk],
                  [AC_SUBST([HAVE_TMPFS], [/Volumes/RAMDisk/bifrost])])
  fi
 
  if test "$HAVE_TMPFS" = "/tmp"; then
    AC_CHECK_FILE([/tmp],
                  [AC_SUBST([HAVE_TMPFS], [/tmp])])
    AC_MSG_WARN([$HAVE_TMPFS may have performance problems for logging])
    AC_SUBST([HAVE_TMPFS], [/tmp/bifrost])
  fi
])
