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
   local startdefault=0
   local durationdefault=10
   local ipathdefault=${MOVIE:-/Users/blyth/Movies/2018_10.mp4}

   local start=${1:-$startdefault}
   local duration=${2:-$durationdefault}
   local ipath=${3:-$ipathdefault}

   start=$(trim-parsetime $start) 

   local idir=$(dirname $ipath)    
   local iname=$(basename $ipath)    
   local istem=${iname/%.*}
   local iext=${iname##*.}

   local opathdefault=$idir/${istem}_trim_${start}_${duration}.${iext}
   local opath=${4:-$opathdefault}
   #[ ! -f "$opath" ] && touch $opath   # now done from the applescript
   local cmd="osascript $applescript $start $duration $ipath $opath"
   echo $msg $cmd
   eval $cmd
}


if [ "$0" != "$BASH_SOURCE" ]; then 
    echo $msg source-ing detected, trim bash function is available for use   
    type trim
else
    trim $*
fi 


