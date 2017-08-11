#!/bin/bash
mkdir -p with_bifrost/blocks
mkdir -p without_bifrost/src
for file in $WITH_BIFROST; do
    if [ ! -f with_bifrost/$file ]; then
        wget https://raw.githubusercontent.com/telegraphic/bunyip/master/$file -O with_bifrost/$file; \
    fi
done
for file in $WITHOUT_BIFROST; do
    if [ ! -f without_bifrost/src/$file ]; then
        wget https://raw.githubusercontent.com/UCBerkeleySETI/gbt_seti/master/src/$file -O without_bifrost/src/$file; \
    fi
done
mkdir -p without_bifrost
