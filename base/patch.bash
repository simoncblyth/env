patch-test(){

   #
   #   1) expands the ball into a patched folder
   #   2) applies the patch
   #   3) compares the patched with the modified 
   #         ... if use same exclusions should be no difference
   #

   local pwd=$PWD
   local msg="=== $FUNCNAME :" 
   
   local def_ball="/usr/local/dyb/trunk_dbg/external/OpenScientist/src/osc_source_v16r1.zip" 
   local def_top="OpenScientist"           ## no simple way to extract this from a ball ?
   local def_patch=/usr/local/dyb/trunk_dbg/installation/trunk/dybinst/patches/osc_source_v16r1.zip.patch

   local ball=${1:-$def_ball}
   [ ! -f "$ball" ] && echo === $FUNCNAME no ball at path $ball && return 1
   
   local dir=$(dirname $ball)
   local name=$(basename $ball)
   local base=${name/.*}
   local type=${name//*./}
   
   cd $dir
   
   local top=${2:-$def_top}
   [ ! -d $top ]   && echo $msg no top $top directory the ball must be expanded first && return 1
   
   local patch=${3:-$def_patch}
   [ ! -f $patch ] && echo $msg no patch file $patch && return 1 
   
   local opt=${4:-$def_opt}
   
   
   local tdir="patched"
   local prior="$tdir/$top"
   
   [ -d "$prior"  ] && echo $msg prior folder $prior exists cannot continue delete this and rerun && return 1 
   [ "$type" == "zip" ] && test ! -d "$prior" && echo $msg unzipping $ball into $prior &&  unzip -d $tdir $ball 
   
   cd $tdir
   echo $msg patching with $patch && patch -p1 < $patch
   
   cd $dir
   echo $msg comparing the modified with the $tdir with opt $opt && diff $opt modified $tdir
   

}

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
   local def_opt="-r --brief"
   
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
   
   local msg="# === $FUNCNAME :"
   echo $msg see  http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/MakingPatches
   echo $msg ball $ball
   echo $msg top  $top   
   echo $msg dir $dir 
   echo $msg name $name 
   echo $msg base $base 
   echo $msg type $type
   echo $msg opt $opt
   
   [ "$type" == "zip" ] && test ! -d original && unzip -d original $ball 
	
	test ! -d modified && mkdir modified 
	test ! -h modified/$top && cd modified && ln -s ../$top $top && cd ..
	test -d modified && test -d original && diff $opt original modified

    cd $pwd

}