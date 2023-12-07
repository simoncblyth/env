#!/bin/bash -l 

usage(){ cat << EOU
epub.sh 
==========

This script is intended to be sourced after the CAP_ environment has been setup

/Users/blyth/.opticks/ntds3/G4CXOpticks/G4CXSimtraceTest/ALL/figs/gxt/mpcap/J003_DownXZ1000.png 


EOU
}


epub_relative_stem()
{
   : when path argument is identified as within one of a few known output folders
   : returns a path relative to the corresponding base directory  

   local path=$1
   local dotopticks=$HOME/.opticks/
   local tmpcache=/tmp/$USER/opticks/
   local tmpdata=/data/$USER/opticks/ 
   local u4mesh=/tmp/U4Mesh_test2/figs/

   local rel  
   case $path in 
      ${dotopticks}*)  rel=${path/$dotopticks/} ;;
      ${tmpcache}*)    rel=${path/$tmpcache/} ;;
      ${tmpdata}*)     rel=${path/$tmpdata/} ;;
      ${u4mesh}*)      rel=${path/$u4mesh/}  ;; 
   esac 

   rel=${rel/\.jpg}
   rel=${rel/\.png}
   echo $rel 
}

epub_pub()
{
    local msg="$FUNCNAME :"
    local cap_path=$1
    local cap_ext=$2
    local rel_stem=$(epub_relative_stem ${cap_path})

    if [ "$PUB" == "1" ]; then 
        local extra=""    ## use PUB=1 to debug the paths 
    else
        local extra="_${PUB}" 
    fi  

    local s5p=/env/presentation/${rel_stem}${extra}${cap_ext}
    local pub=$HOME/simoncblyth.bitbucket.io$s5p
    local s5p_line="$s5p 1280px_720px"

    local vars="0 BASH_SOURCE FUNCNAME cap_path cap_ext rel_stem PUB extra s5p pub s5p_line"
    for var in $vars ; do printf "%20s : %s\n" $var "${!var}" ; done  

    mkdir -p $(dirname $pub)

    if [ "$PUB" == "" ]; then 
        echo $msg skipping copy : to do the copy you must set PUB to some descriptive string 
    elif [ "$PUB" == "1" ]; then 
        echo $msg skipping copy : to do the copy you must set PUB to some descriptive string 

    elif [ "$PUB" == "1" ]; then 
        echo $msg skipping copy : to do the copy you must set PUB to some descriptive string 
    elif [ -f "$pub" ]; then
        echo $msg published path exists already : NOT COPYING : delete it or set PUB to some different extra string to distinguish the name 
        echo $msg skipping copy : to do the copy you must set PUB to some descriptive string rather than just using PUB=1
    else
        echo $msg copying cap_path to pub 
        cp $cap_path $pub
        echo $msg add s5p_line to s5_background_image.txt
    fi
}


vars="BASH_SOURCE CAP_PATH CAP_EXT"
for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 

epub_pub $CAP_PATH $CAP_EXT


