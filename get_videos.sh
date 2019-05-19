#!/bin/bash
function usage {
    echo "usage: $0 25|26."
    exit 1
}

if [ $# -eq 1 ]; then
    if [ $1 -eq 25 ]; then
        echo "Copying files from robotica@192.168.1.25..."
        rsync -ram -e ssh --include '*/' --include '*.avi' --exclude '*' robotica@192.168.1.25:myDir ./resources/videos
    elif [ $1 -eq 26 ]; then
        echo "Copying files from robotica@192.168.1.26..."
        rsync -ram -e ssh --include '*/' --include '*.avi' --exclude '*' robotica@192.168.1.26:myDir ./resources/videos
    else
        usage
    fi
else
    usage
fi