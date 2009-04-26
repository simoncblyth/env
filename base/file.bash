file-src(){ echo base/file.bash ; }
file-source(){ echo ${BASH_SOURCE:-$(env-home)/$(file-src)} ; }
file-vi(){     vi $(file-source) ; }
file-env(){

   local msg="=== $FUNCNAME :"
}


file-usage(){

   cat << EOU
   
   
     file-exts path-or-name
     
        Returns list of extensions strings (excluding .mine and .merge)
        to the given name-or-path for existing files eg, if   
          
          ls glmodule.c.*
             glmodule.c.mine 
             glmodule.c.r790 
            glmodule.c.r833
   
          echo $(file-exts glmodule.c)
             r790 r833
      
EOU

}





#
#  put/get to same place in file heirarchy on remote node, assuming it exists
#
#   file-rpwd
#   file-p
#   file-g
#
#
#   idea ...  have LOCAL_BASE for all nodes , so can create a LOCAL_BASE relative copy 
#
#

file-rpwd(){     echo $PWD | perl -p -e 's|$ENV{"HOME"}/(.*)|$1|' ;  } ## returns the path relative to home
file-p(){ [ -f "$1" ] && scp $1  ${2:-$TARGET_TAG}:$(file-rpwd)/  || echo need at least one argument ; }
file-g(){ [ -z "$1" ] || scp ${2:-$TARGET_TAG}:$(file-rpwd)/$1 .  || echo need at least one argument ; }



file-dirlist(){
 
 local iwd=$(pwd)
 local base=${1:-dummy}

 for item in $base/*
 do
    if [ -d $item ]; then
	  local rela=$(basename $item)	
      echo $rela
    fi		
 done	 
  

}


file-package-topdir(){
   #
   #  return the top directories present in a tar.gz or zip , 
   #  if only one such directory return with success status, otherwise return with error status
   #
   local pkg=${1:-dummy}
   [ -f "$pkg"  ] ||  return 1 
   
   local cmd 
   
   if [ "${pkg:(-4):4}" == ".zip" ]; then
   
       cmd="unzip -l $pkg | perl -n -e 'BEGIN{ @n=();}; m|\s*\d*\s*[\d-]*\s*[\d:]*\s*(\S*?)/(\S*)\s*| && do { push(@n,\$1) if(grep(\$1 eq \$_,@n)==0); } ; END{ print \"\$_\\n\" for(@n);} '"
       
   elif ([ "${pkg:(-7):7}" == ".tar.gz" ] || [ "${pkg:(-4):4}" == ".tgz" ]) then
   
       cmd="tar -ztf $pkg | perl -n -e 'BEGIN{ @n=();}; m|(.*?)/.*| && do { push(@n,\$1) if(grep(\$1 eq \$_,@n)==0); } ; END{ print \"\$_\\n\" for(@n);}  '"
   
   else
      return 1
   fi

   #echo $cmd
   eval $cmd 
}

file-diff(){

   #  opendiff is a Mac OS X commandline interface to the FileMerge GUI application 
   #
   #  NB note this requires the path passed is within the current DYW, otherwise will fail with no such path 
   #
   
   
   test -d "$DYW" || ( echo variable DYW $DYW does not point to a folder && return 1 )
   
   local rel=${1:-dummy}
   local src=${2:-blyth-optical}  ## source branch
   local tgt=${3:-dywcvs}    ## target branch 
   local anc=${4:-trunk}    ## the ancestor allows conflicts to be highlighted in red
   local mer=${5:-$tgt}   ## branch for the resulting merged file
   
   local lhs=$src
   local rhs=$tgt
    
   local abspath=$PWD/$rel 
    
   echo ==== file-diff  rel $rel src $src tgt $tgt anc $anc mer $mer ==== 
    
   [ -f "$abspath" ] || ( echo no such path $abspath && return 1 )
   
   local path=${abspath#$DYW/}   ## make the absolute path relative to DYW
   local iwd=$(pwd)
   
   local cmd="cd $DYW/.. && opendiff $lhs/$path $rhs/$path -ancestor $anc/$path -merge $mer/$path.filemerge "
   echo $cmd
   eval $cmd

   cd $iwd 

}



file-exts(){

  local msg="=== $FUNCNAME :"
  local arg=$1
  local iwd=$PWD
  local dir=$(dirname $arg) 
  
  #echo $msg $dir  
  cd $dir
  local namebase=$(basename $arg)
  local f
  local t
  for n in $(ls -1 $namebase.*) ; do
     t=${n/$namebase./}
     if [ "$t" == "mine" -o "$t" == "merge" ]; then
       echo -n
     else
       echo $t     
     fi
  done
  cd $iwd
}





file-merge(){

   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   
   local path=$1
   local mine="$path.mine"
   
   [ ! -f "$mine" ] && echo $msg no .mine file at: $mine ... abort && return 1 
   
   local ext=$(file-exts $path)
   local ancestor=$(file-first $ext)
   local head=$(file-second $ext)
   
   [ ! -f "$path.$head" ]     && echo $msg no     head file $path.$head      ... abort && return 1
   [ ! -f "$path.$ancestor" ] && echo $msg no ancestor file $path.$ancestor  ... abort && return 1
   
   local cmd="opendiff $path.mine $path.$head -ancestor $path.$ancestor -merge $path.merge"


   echo $msg comparing divergences from a common ancestor  $path.$ancestor between local changes $path.mine and repository changes $path.$head
   echo $msg use arrow keys to pick which mods relative to the common ancestor to use for the merged file 
   echo $msg $cmd
   
   eval $cmd

}


file-first(){ echo $1 ; }
file-second(){ echo $2 ; }


#file-first(){
#  for arg in $*
#  do
#     echo $arg 
#     return 0
#  done
#}

file-size-lt(){

  local path=$1 
  local ksize=${2:-100}
  local duk=$(du -k $path) 
  local size=$(file-first $duk)

  [ $size -lt $ksize ] && return 0 || return 1

}


file-testcp(){

  #
  #   without arguments :
  #         creates a file of 100Mb and copies it, timing the action
  #
  #   with argument (note no tailing slash):
  #          file-testcp /tmp/tt
  #          file-testcp /Volumes/Hello 
  #
  #              copies the test file to the provided directory path (that is assumed to exist) 
  #

  local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
  local dir=${1:-$tmp} 
  local msg="=== $FUNCNAME :"
  echo $msg testing speed to copy to $dir
  python $(env-home)/base/test-cp.py  $dir
   
}


