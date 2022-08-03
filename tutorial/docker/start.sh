#!/bin/bash
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

set -e

# Exec the specified command or fall back on bash
if [ $# -eq 0 ]; then
    cmd=( "bash" )
else
    cmd=( "$@" )
fi

run-hooks () {
    # Source scripts or run executable files in a directory
    if [[ ! -d "$1" ]] ; then
        return
    fi
    echo "$0: running hooks in $1"
    for f in "$1/"*; do
        case "$f" in
            *.sh)
                echo "$0: running $f"
                source "$f"
                ;;
            *)
                if [[ -x "$f" ]] ; then
                    echo "$0: running $f"
                    "$f"
                else
                    echo "$0: ignoring $f"
                fi
                ;;
        esac
    done
    echo "$0: done running hooks in $1"
}

run-hooks /usr/local/bin/start-notebook.d

# Handle special flags if we're root
if [ $(id -u) == 0 ] ; then

    # Only attempt to change the lwa username if it exists
    if id lwa &> /dev/null ; then
        echo "Set username to: $LSL_USER"
        usermod -d /home/$LSL_USER -l $LSL_USER lwa
    fi

    # Handle case where provisioned storage does not have the correct permissions by default
    # Ex: default NFS/EFS (no auto-uid/gid)
    if [[ "$CHOWN_HOME" == "1" || "$CHOWN_HOME" == 'yes' ]]; then
        echo "Changing ownership of /home/$LSL_USER to $LSL_UID:$LSL_GID with options '${CHOWN_HOME_OPTS}'"
        chown $CHOWN_HOME_OPTS $LSL_UID:$LSL_GID /home/$LSL_USER
    fi
    if [ ! -z "$CHOWN_EXTRA" ]; then
        for extra_dir in $(echo $CHOWN_EXTRA | tr ',' ' '); do
            echo "Changing ownership of ${extra_dir} to $LSL_UID:$LSL_GID with options '${CHOWN_EXTRA_OPTS}'"
            chown $CHOWN_EXTRA_OPTS $LSL_UID:$LSL_GID $extra_dir
        done
    fi

    # handle home and working directory if the username changed
    if [[ "$LSL_USER" != "lwa" ]]; then
        # changing username, make sure homedir exists
        # (it could be mounted, and we shouldn't create it if it already exists)
        if [[ ! -e "/home/$LSL_USER" ]]; then
            echo "Relocating home dir to /home/$LSL_USER"
            mv /home/lwa "/home/$LSL_USER" || ln -s /home/lwa "/home/$LSL_USER"
        fi
        # if workdir is in /home/lwa, cd to /home/$LSL_USER
        if [[ "$PWD/" == "/home/lwa/"* ]]; then
            newcwd="/home/$LSL_USER/${PWD:13}"
            echo "Setting CWD to $newcwd"
            cd "$newcwd"
        fi
    fi

    # Change UID:GID of LSL_USER to LSL_UID:LSL_GID if it does not match
    if [ "$LSL_UID" != $(id -u $LSL_USER) ] || [ "$LSL_GID" != $(id -g $LSL_USER) ]; then
        echo "Set user $LSL_USER UID:GID to: $LSL_UID:$LSL_GID"
        if [ "$LSL_GID" != $(id -g $LSL_USER) ]; then
            groupadd -g $LSL_GID -o ${LSL_GROUP:-${LSL_USER}}
        fi
        userdel $LSL_USER
        useradd --home /home/$LSL_USER -u $LSL_UID -g $LSL_GID -G 100 -l $LSL_USER
    fi

    # Enable sudo if requested
    if [[ "$GRANT_SUDO" == "1" || "$GRANT_SUDO" == 'yes' ]]; then
        echo "Granting $LSL_USER sudo access and appending $CONDA_DIR/bin to sudo PATH"
        echo "$LSL_USER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/notebook
    fi

    # Add $CONDA_DIR/bin to sudo secure_path
    sed -r "s#Defaults\s+secure_path\s*=\s*\"?([^\"]+)\"?#Defaults secure_path=\"\1:$CONDA_DIR/bin\"#" /etc/sudoers | grep secure_path > /etc/sudoers.d/path

    # Exec the command as LSL_USER with the PATH and the rest of
    # the environment preserved
    run-hooks /usr/local/bin/before-notebook.d
    echo "Executing the command: ${cmd[@]}"
    exec sudo -E -H -u $LSL_USER PATH=$PATH XDG_CACHE_HOME=/home/$LSL_USER/.cache PYTHONPATH=${PYTHONPATH:-} "${cmd[@]}"
else
    if [[ "$LSL_UID" == "$(id -u lwa)" && "$LSL_GID" == "$(id -g lwa)" ]]; then
        # User is not attempting to override user/group via environment
        # variables, but they could still have overridden the uid/gid that
        # container runs as. Check that the user has an entry in the passwd
        # file and if not add an entry.
        STATUS=0 && whoami &> /dev/null || STATUS=$? && true
        if [[ "$STATUS" != "0" ]]; then
            if [[ -w /etc/passwd ]]; then
                echo "Adding passwd file entry for $(id -u)"
                cat /etc/passwd | sed -e "s/^lwa:/nayvoj:/" > /tmp/passwd
                echo "lwa:x:$(id -u):$(id -g):,,,:/home/lwa:/bin/bash" >> /tmp/passwd
                cat /tmp/passwd > /etc/passwd
                rm /tmp/passwd
            else
                echo 'Container must be run with group "root" to update passwd file'
            fi
        fi

        # Warn if the user isn't going to be able to write files to $HOME.
        if [[ ! -w /home/lwa ]]; then
            echo 'Container must be run with group "users" to update files'
        fi
    else
        # Warn if looks like user want to override uid/gid but hasn't
        # run the container as root.
        if [[ ! -z "$LSL_UID" && "$LSL_UID" != "$(id -u)" ]]; then
            echo 'Container must be run as root to set $LSL_UID'
        fi
        if [[ ! -z "$LSL_GID" && "$LSL_GID" != "$(id -g)" ]]; then
            echo 'Container must be run as root to set $LSL_GID'
        fi
    fi

    # Warn if looks like user want to run in sudo mode but hasn't run
    # the container as root.
    if [[ "$GRANT_SUDO" == "1" || "$GRANT_SUDO" == 'yes' ]]; then
        echo 'Container must be run as root to grant sudo permissions'
    fi

    # Execute the command
    run-hooks /usr/local/bin/before-notebook.d
    echo "Executing the command: ${cmd[@]}"
    exec "${cmd[@]}"
fi
