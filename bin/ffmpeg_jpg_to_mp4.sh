#!/bin/bash -l 

usage(){ cat << EOU

ffmpeg_jpg_to_mp4.sh
=======================

::

   cd /directory/containing/the/jpg
   ffmpeg_jpg_to_mp4.sh FlightPath 

EOU 
}


env-
ffmpeg-
ffmpeg-export

msg="=== $0 :"

pfx=${1:-FlightPath}
jpg=${pfx}?????.jpg
mp4=${pfx}.mp4

rm -f ${mp4}
   
ls -alst $jpg

echo $msg pfx $pfx creating mp4 $mp4 from jpg $jpg 

ffmpeg -i ${pfx}%05d.jpg $mp4 && rm ${pfx}?????.jpg

ls -l $mp4 
du -h $mp4

