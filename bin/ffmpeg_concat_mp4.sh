#!/bin/bash -l 

usage(){ cat << EOU
ffmpeg_concat_mp4.sh
=======================

::

   cd ~/Movies
   OUT=987.mp4 ffmpeg_concat_mp4.sh lFasteners_phys_9,_256.mp4 lFasteners_phys_8,_256.mp4 lFasteners_phys_7,_256.mp4

EOU
}

env-
ffmpeg-
ffmpeg-export

msg="=== $0 :"
out=${OUT:-combined.mp4}
txt=${out/.mp4}.txt 

echo $msg out $out txt $txt 

pwd
ls -alst $* 

if [ -f "$txt" ]; then 
    rm $txt
fi 
for mp4 in $* ; do
    printf "file '$mp4'\n" >> $txt
done 
cat $txt 

cmd="ffmpeg -f concat -safe 0 -i $txt -c copy $out"
echo $cmd 
eval $cmd 

