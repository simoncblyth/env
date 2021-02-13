#!/bin/bash 
msg="$BASH_SOURCE :"
trim-vi(){ vi $BASH_SOURCE && source $BASH_SOURCE ; }
trim-(){   source $BASH_SOURCE ; }
trim-usage(){ cat << EOU
trim.sh 
========

Script usage mode::

  trim.sh 42     
  trim.sh 42 5 

  trim.sh 02:14 

Sourced usage mode::

   source ~/env/bin/trim.sh 
   type trim 
   trim 02:14 10 /path/to/input.mp4 

EOU
}

trim-applescript(){
   local dir=$(dirname $BASH_SOURCE)
   local name=$(basename $BASH_SOURCE)
   local stem=${name/%.*}
   local applescript=$dir/$stem.applescript
   echo $applescript 
}

trim-parsetime(){
   local arg=$1
   if [ "${arg/:}" == "$arg" ]; then
       local sec=$arg
   else
       local bef=${arg/:*}
       local aft=${arg#*:}
       local sec=$(( $bef * 60 + $aft ))
   fi  
   echo $sec
}

trim()
{
   : prepares arguments of quicktime trim applescript 
   local msg="=== $FUNCNAME :"
   local applescript=$(trim-applescript)
   local starttimedefault=0
   local durationdefault=10
   local ipathdefault=${MOVIE:-/Users/blyth/Movies/opticks201810.mp4}

   local starttime=${1:-$starttimedefault}
   local duration=${2:-$durationdefault}
   local ipath=${3:-$ipathdefault}

   local startseconds=$(trim-parsetime $starttime) 
   local idir=$(dirname $ipath)    
   local iname=$(basename $ipath)    
   local istem=${iname/%.*}
   local iext=${iname##*.}

   local opathdefault=$idir/${istem}_trim_${starttime/:}_${duration}.${iext}
   local opath=${4:-$opathdefault}

   echo $msg starttime $starttime duration $duration ipath $ipath 
   echo $msg startseconds $startseconds opath $opath  

   if [ -f "$opath" ]; then
       echo $msg opath $opath exists already 
   else
       local cmd="osascript $applescript $startseconds $duration $ipath $opath"
       echo $msg $cmd
       eval $cmd
   fi 

   local ans
   read -p "$msg press return to open it or anything else to skip " ans 
   [ "$ans" != "" ] && echo $msg skip && return 0  
   open $opath 
   return 0 
}


if [ "$0" != "$BASH_SOURCE" ]; then 
    echo $msg source-ing detected, trim bash function is available for use   
    type trim
else
    trim $*
fi 


