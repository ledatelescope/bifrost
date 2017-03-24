#!/bin/bash

#
# Simple script to help kill off a bifrost process and clean up afterwards
#

# Pass everything to kill
kill $@

# Did it work?
if [[ "$?" -eq "0" ]]; then
	## Pull out the PID from the argument list (the last one)
	for pid; do true; done

	## Clean out /dev/shm/bifrost if the process is really dead
	if [[ ! -d /proc/${pid} ]]; then
		rm -rf /dev/shm/bifrost/${pid}
	fi
else
	## Nope, return the exit code of 'kill'
	exit $?
fi

