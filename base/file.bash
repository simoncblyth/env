
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


ini-edit(){
   ## utility for editing INI files 
   perl -e 'require "$ENV{'HOME'}/$ENV{'ENV_BASE'}/base/INI.pm" ; &INI::EDIT(@ARGV) ; ' $*
}



