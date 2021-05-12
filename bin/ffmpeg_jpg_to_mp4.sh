#!/bin/bash -l 

usage(){ cat << EOU
ffmpeg_jpg_to_mp4.sh
=======================

The .jpg are deleted after the .mp4 is created from them::

   cd /directory/containing/the/jpg

   ffmpeg_jpg_to_mp4.sh

The jpg must be named with a 5 digit zero filled index:: 

    some_name_prefix_00000.jpg 


EOU
}


pwd

env-
ffmpeg-
ffmpeg-export

msg="=== $0 :"

pwd
jpg0=$(ls -1 *00000.jpg)
[ -z "$jpg0" ] && echo $msg FAILED to find render named pfx00000.jpg && exit 1 

pfx=${jpg0/00000.jpg}
jpg=${pfx}?????.jpg
mp4=${pfx}.mp4
first=${pfx}00000.jpg

echo $msg jpg0 $jpg0 pfx $pfx mp4 $mp4 first $first
[ "$first" != "$jpg0" ] && echo UNEXPECTED first $first jpg0 $jpg0 && exit 2 

rm -f ${mp4}
ls -alst $jpg

if [ -f "$first" ]; then
    cp $first ${first/.jpg}.jpeg  # copy first frame to .jpeg extension so it doesnt get deleted 
fi 

ffmpeg -i ${pfx}%05d.jpg $mp4 && rm ${pfx}?????.jpg

ls -l $mp4 
du -h $mp4

pwd
exit 0
