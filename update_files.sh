#!/bin/bash
function usage {
    echo "usage: $0 25|26."
    exit 1
}

if [ $# -eq 1 ]; then
    if [ $1 -eq 25 ]; then
        echo "Copying files to robotica@192.168.1.25..."
        rsync -ram -e 'sshpass -p upmRobotica ssh' --include '*/' --include '*.py' --include '*.png' --include '*.jpg' --exclude '*' . robotica@192.168.1.25:myDir
        rsync -ram -e 'sshpass -p upmRobotica ssh' --include '*/' --include '*.avi' --exclude '*' robotica@192.168.1.25:myDir/resources/videos ./resources/videos
    elif [ $1 -eq 26 ]; then
        echo "Copying files to robotica@192.168.1.26..."
        rsync -ram -e 'sshpass -p upmRobotica ssh' --include '*/' --include '*.py' --include '*.png' --include '*.jpg' --exclude '*' . robotica@192.168.1.26:myDir
        rsync -ram -e 'sshpass -p upmRobotica ssh' --include '*/' --include '*.avi' --exclude '*' robotica@192.168.1.26:myDir/resources/videos ./resources/videos    
    else
        usage
    fi
else
    usage
fi