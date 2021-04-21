#!/bin/bash -l 

usage(){ cat << EOU
ffmpeg_jpg_to_mp4.sh
=======================

The .jpg are deleted after the .mp4 is created from them::

   cd /directory/containing/the/jpg

   ffmpeg_jpg_to_mp4.sh FlightPath 

EOU
}


pwd

env-
ffmpeg-
ffmpeg-export

msg="=== $0 :"

pfx=${1:-FlightPath}
jpg=${pfx}?????.jpg
mp4=${pfx}.mp4


pwd
rm -f ${mp4}
ls -alst $jpg

echo $msg pfx $pfx creating mp4 $mp4 from jpg $jpg 


first=${pfx}00000.jpg

if [ -f "$first" ]; then
    cp $first ${first/.jpg}.jpeg  # copy first frame to .jpeg extension so it doesnt get deleted 
fi 

ffmpeg -i ${pfx}%05d.jpg $mp4 && rm ${pfx}?????.jpg

ls -l $mp4 
du -h $mp4

