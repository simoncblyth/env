
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

   ##  opendiff is a Mac OS X commandline interface to the FileMerge GUI application 
   
   # simplify file-diff arguments via defaults , ie can just specifiy a source and target branch together with the 
   
   
   test -d "$DYW" || ( echo variable DYW $DYW does not point to a folder && return 1 )
   
   local rel=${1:-dummy}
   local src=${2:-blyth-optical}
   local tgt=${3:-dywcvs}
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





