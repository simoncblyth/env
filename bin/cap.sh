#!/bin/bash -l 
usage(){ cat << EOU
cap.sh
==========

1. Arrange a very thin terminal window placed at the bottom of the 
   screen from which to launch the capture.


2. Run the pyvista using script 

3. Invoke the capture script off the PATH as usually ~/env/bin is in PATH::
   
   pvcap.sh 
   mpcap.sh 
   sfcap.sh 
   cap.sh 

All those are symbolically linked to cap.sh and change the parameters of
the crop. When running the script:

1. the terminal window will turn blue
2. select the desired window to capture and within 2 seconds make sure to 
   make it the frontmost window
3. after 2 seconds the screen capture sound should be audible and
   the captured png is cropped 

Envvars control the directiory and name of screen captures.

CAP_DIR
CAP_STEM 

EOU
}

SCRIPT=$(basename $BASH_SOURCE)
style=safari
case $SCRIPT in 
   pvcap.sh) style=pyvista ;;
   mpcap.sh) style=matplotlib ;;
   sfcap.sh) style=safari ;;
esac
stem=${SCRIPT/.sh}

CAP_DIR=${CAP_DIR:-/tmp/$USER/opticks/cap}
CAP_STEM=${CAP_STEM:-$stem}
path=$CAP_DIR/${CAP_STEM}.png
uncropped=$CAP_DIR/${CAP_STEM}_uncropped.png

vars="BASH_SOURCE SCRIPT style stem CAP_DIR CAP_STEM uncropped path"
for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 

mkdir -p $(dirname $path)
screencapture -T 2 -i -w -W -o -S $uncropped

${IPYTHON:-ipython} ~/env/doc/crop.py -- --style $style --replace $uncropped 

ls -l $path $uncropped
open $path

