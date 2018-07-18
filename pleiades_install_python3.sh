#!/usr/bin/env bash

PACKAGES=(
    https://www.rpmfind.net/linux/fedora/linux/development/rawhide/Everything/x86_64/os/Packages/g/glibc-2.27.9000-27.fc29.x86_64.rpm
    https://www.rpmfind.net/linux/fedora/linux/development/rawhide/Everything/x86_64/os/Packages/p/python3-libs-3.6.5-4.fc29.x86_64.rpm
    https://www.rpmfind.net/linux/fedora/linux/development/rawhide/Everything/x86_64/os/Packages/p/python3-3.6.5-4.fc29.x86_64.rpm
)

for package in ${PACKAGES[@]}; do
    FILENAME=${package##*/}
    FILENAME_CPIO=${FILENAME}.cpio
    # Get package if we don't already have it
    if [ ! -f ${FILENAME} ]; then
        wget ${package}
    fi

    rpm2cpio ${FILENAME} > ${FILENAME_CPIO}
    cpio -idv < ${FILENAME_CPIO}
done

#export LD_LIBRARY_PATH=/home/sfoster/python3/usr/lib64:/home/sfoster/python3/lib64