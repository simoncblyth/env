

patch-diff(){

   # 
   # allows a modified expanded ball to be compared with the original ball content
   # as described at 
   #      http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/MakingPatches
   #  
   #  
   #    patch-diff /usr/local/dyb/trunk_dbg/external/OpenScientist/src/osc_source_v16r1.zip OpenScientist "--brief -x "*.sh" -x "*.csh" -x "*.bat" -x foreign -x sh -x bin_obuild " 
   #
   #
   #

   local pwd=$PWD
   
   local def_ball="/usr/local/dyb/trunk_dbg/external/OpenScientist/src/osc_source_v16r1.zip" 
   local def_top="OpenScientist"   ## no simple way to extract this from a ball ?
   local def_opt="--brief"
   
   local ball=${1:-$def_ball}
   [ ! -f "$ball" ] && echo === $FUNCNAME no ball at path $ball && return 1
   
   local top=${2:-$def_top}
   local opt=${3:-$def_opt}
   
   local dir=$(dirname $ball)
   local name=$(basename $ball)
   local base=${name/.*}
   local type=${name//*./}
   
   cd $dir
   [ ! -d $top ] && echo === $FUNCNAME no top $top directory the ball must be expanded first && return 1
   
   echo === $FUNCNAME ball $ball   
   echo === $FUNCNAME dir $dir name $name base $base type $type top $top opt $opt
   
   [ "$type" == "zip" ] && test ! -d original && unzip -d original $ball 
	
	test ! -d modified && mkdir modified 
	test ! -h modified/$top && cd modified && ln -s ../$top $top && cd ..
	test -d modified && test -d original && diff -r $opt original modified

    cd $pwd

}