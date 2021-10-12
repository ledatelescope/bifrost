AC_DEFUN([AX_CHECK_TMPFS],
[
  AC_PROVIDE([AX_CHECK_TMPFS])
 
  AC_SUBST([HAVE_TMPFS], [/tmp])
  
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
  
  CPPFLAGS="$CPPFLAGS -DBF_PROCLOG_DIR='\"$HAVE_TMPFS\"'"
])
