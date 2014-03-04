patch-usage(){ cat << EOU
::

    [blyth@belle7 patches]$ pwd
    /data1/env/local/dyb/external/build/LCG/g4checkpatch/patches

    [blyth@belle7 patches]$ patch-;patch-match geant4.9.2.p01.patch
    === patch-match : truncating geant4.9.2.p01.patch.patch-match : reconstructing a combi patch from single file patches in matched order
    geant4.9.2.p01_source_geometry_solids_Boolean_src_G4SubtractionSolid.cc.patch
    geant4.9.2.p01_source_processes_electromagnetic_lowenergy_src_G4hLowEnergyLoss.cc.patch
    geant4.9.2.p01_source_processes_hadronic_processes_include_G4ElectronNuclearProcess.hh.patch
    geant4.9.2.p01_source_processes_hadronic_processes_include_G4PhotoNuclearProcess.hh.patch
    geant4.9.2.p01_source_processes_hadronic_processes_include_G4PositronNuclearProcess.hh.patch
    geant4.9.2.p01_source_processes_hadronic_processes_src_G4ElectronNuclearProcess.cc.patch
    geant4.9.2.p01_source_processes_hadronic_processes_src_G4PhotoNuclearProcess.cc.patch
    geant4.9.1.p01_source_processes_optical_include_G4OpBoundaryProcess.hh.patch
    geant4.9.2.p01_source_materials_include_G4MaterialPropertyVector.hh.patch
    geant4.9.2.p01_source_materials_src_G4MaterialPropertiesTable.cc.patch
    geant4.9.2.p01_source_materials_src_G4MaterialPropertyVector.cc.patch

Huh this is the new dyb, without my mods... how this diff between the order matched patches ?::

    [blyth@belle7 patches]$ diff geant4.9.2.p01.patch geant4.9.2.p01.patch.patch-match 
    ...
    < diff -u -r geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc geant4.9.2.p01/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc
    < --- geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc      2009-03-16 10:06:45.000000000 -0400
    < +++ geant4.9.2.p01/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc   2009-06-11 01:22:12.000000000 -0400
    ---
    > --- geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc      2009-03-16 22:06:45.000000000 +0800
    > +++ geant4.9.2.p01/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc   2014-03-04 18:04:57.000000000 +0800
    168,200c162,163
    < --- geant4.9.1.p01/source/processes/optical/include/G4OpBoundaryProcess.hh    2007-10-15 17:16:24.000000000 -0400
    < +++ geant4.9.1.p01/source/processes/optical/include/G4OpBoundaryProcess.hh    2009-04-14 11:28:42.000000000 -0400
    < @@ -408,10 +408,23 @@
    <  void G4OpBoundaryProcess::DoReflection()
    <  {
    <          if ( theStatus == LambertianReflection ) {
    < -
    < -          NewMomentum = G4LambertianRand(theGlobalNormal);
    < -          theFacetNormal = (NewMomentum - OldMomentum).unit();
    < -
    < +       // wangzhe
    < +       // Original:
    < +          //NewMomentum = G4LambertianRand(theGlobalNormal);
    < +          //theFacetNormal = (NewMomentum - OldMomentum).unit();
    < +       
    < +       // Temp Fix:
    < +       if(theGlobalNormal.mag()==0) {
    < +            std::cout<<"Error. Zero caught. A normal vector with mag be 0. May trigger a infinite loop later."<<std::endl;
    < +            std::cout<<"A temporary solution: Photon is forced to go back along its original path."<<std::endl;
    < +            std::cout<<"Test from MDC09a tells the effect of this bug is tiny."<<std::endl;
    < +         G4ThreeVector myVec(0,0,0);
    < +         theFacetNormal = (myVec - OldMomentum).unit();
    < +       } else {
    < +         NewMomentum = G4LambertianRand(theGlobalNormal);
    < +         theFacetNormal = (NewMomentum - OldMomentum).unit();
    < +       }
    < +       // wz
    <          }
    <          else if ( theFinish == ground ) {
    <  


EOU
}

patch-vi(){ vi $BASH_SOURCE ; }


patch-match-line(){
  shift
  local path=$1
  local name=$(echo $path | tr "/" "_").patch
  echo $name
}
patch-match(){
  local line
  local mpatch=$1
  local opatch=$mpatch.$FUNCNAME
  local msg="=== $FUNCNAME :"
  echo $msg truncating $opatch : reconstructing a combi patch from single file patches in matched order
  echo > $opatch
  local fpatch 
  grep -- +++ $mpatch | while read line ; do
      fpatch=$(patch-match-line $line)
      echo $fpatch
      [ -f "$fpatch" ] && cat $fpatch >> $opatch
  done
}


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
   local def_opt="-r --brief"

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
#   [ "$type" == "zip" ] && test ! -d "$prior" && echo $msg unzipping $ball into $prior &&  unzip -d $tdir $ball 
 
   test ! -d "$prior" && echo $msg copying original/$top into $tdir && cp -Rp original/$top $tdir/ 
	 
	     
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
