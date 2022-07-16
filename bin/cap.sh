#!/bin/bash -l 
usage(){ cat << EOU
cap.sh : screen capture with chrome cropping tool 
=====================================================

1. Run the script producing the image to grab

2. Make sure the window to be captured is partially visible behind 
   the Terminal.app window and then invoke the relevant capture script off the PATH 
   as usually ~/env/bin is in PATH, use the one corresponding to the window type.::

   source pvcap.sh ## pyvista
   source mpcap.sh ## matplotlib
   source sfcap.sh ## safari

All those are symbolically linked to cap.sh and change the parameters of
the crop with different crop paramters. When running the script:

Typically this capture script is invoked from higher level scripts that set 
the envvars CAP_BASE CAP_REL CAP_STEM to control location and naming of captures.
For example::

    cx
    ./cxs_debug.sh pvcap 
    ./cxs_debug.sh mpcap 

What the envvars are used for:

CAP_BASE
    directory in which to save the captures, 
    typically this will be $FOLD/figs the default is /tmp/$USER/opticks

CAP_REL
    relative directory beneath CAP_BASE for organization, default "cap",
    for example use "gxt" : the stem of the name of the driving script
    As the captype eg "pvcap" "mpcap" "sfcap" is auto-included
    do not set CAP_REL to those.  

CAP_STEM
    stem of the captured file, default cap_stem_default
    as this stem might appear on presentation pages without the 
    rest of the path it is useful to collect identity infomation 
    but avoid being too long, for example summarize with : ${GEOM}_${IPSTEM} 

CAP_EXT
    extension of the capture file, default ".png", changing to ".jpg" is possible 


Putting these together gives the path of the capture::

    capdir=${CAP_BASE}/${CAP_REL}/${captype}
    upath=${capdir}/${CAP_STEM}_uncropped${CAP_EXT}
    cpath=${capdir}/${CAP_STEM}${CAP_EXT}



On running the capture script:

1. the invoking terminal window will turn blue
2. select the desired window to capture and make sure within 2 seconds
   to make it the frontmost window with no obscuring other windows
3. after 2 seconds the screen capture sound should be audible and
   the captured png is cropped and opened in Preview

  
For onward use of the captures use argument env to just define environment without 
doing the capture::

   source pvcap.sh env ## pyvista
   source mpcap.sh env ## matplotlib
   source sfcap.sh env ## safari

EOU
}

cap_arg=${1:-cap}
SCRIPT=$(basename $BASH_SOURCE)
style=safari
case $SCRIPT in 
   pvcap.sh) style=pyvista ;;
   mpcap.sh) style=matplotlib ;;
   sfcap.sh) style=safari ;;
     cap.sh) style=generic ;;  
esac
captype=${SCRIPT/.sh}

CAP_BASE=${CAP_BASE:-/tmp/$USER/opticks}
CAP_REL=${CAP_REL:-cap}
CAP_STEM=${CAP_STEM:-cap_stem_default}
CAP_EXT=".png"

capdir=${CAP_BASE}/${CAP_REL}/${captype}
upath=${capdir}/${CAP_STEM}_uncropped${CAP_EXT}
cpath=${capdir}/${CAP_STEM}${CAP_EXT}


vars="BASH_SOURCE SCRIPT style stem CAP_BASE CAP_REL CAP_STEM capdir upath cpath"
cap_dumpvars(){ for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done ; }
cap_dumpvars

if [ "${cap_arg}" == "help" ]; then 
   usage
fi 


if [ "${cap_arg}" == "cap" ]; then 

    mkdir -p $(dirname $upath)
    screencapture -T 2 -i -w -W -o -S $upath

    ${IPYTHON:-ipython} ~/env/doc/crop.py -- --style $style --replace $upath

    ls -l $upath $cpath
    open $cpath

elif [ "${cap_arg}" == "open" ]; then 

    ls -l $upath $cpath
    open $cpath

elif [ "${cap_arg}" == "env" ]; then 

    ls -l $upath $cpath
    export CAP_PATH=$cpath
    export CAP_EXT=$CAP_EXT
    vars="cap_arg CAP_BASE CAP_REL CAP_STEM CAP_PATH CAP_EXT"
    cap_dumpvars 

else
    echo $msg cap_arg ${cap_arg} unhandled

fi 

