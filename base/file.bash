
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


file-tgz-topdir(){
   #
   # return the top directories present in a tarball, if only one such directory return with success status, otherwise return with error status
   #
   local tgz=${1:-dummy}
   [ -f "$tgz"  ] ||  return 1 
   tar -ztf  $tgz | perl -n -e 'BEGIN{ @n=(); }; m|(.*?)/.*| && do { push(@n,$1) if(grep($1 eq $_,@n)==0); } ; END{ print "@n " ; exit(1) if($#n + 1 != 1) ;} '
}





