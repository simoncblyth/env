# === proj-gen- : env graphics/openmeshclient/omc OpenMesh fgd graphics/openmeshclient fgp graphics/openmeshclient/omc.bash fgn omc fgh graphics/openmeshclient pkg OpenMesh
omc-rel(){      echo graphics/openmeshclient ; }
omc-src(){      echo $(omc-rel)/omc.bash ; }
omc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(omc-src)} ; }
omc-vi(){       vi $(omc-source) $(omc-sdir)/CMakeLists.txt $(omc-tdir)/CMakeLists.txt ; }
omc-usage(){ cat << EOU


EOU
}

omc-env(){  
   elocal- 
   opticks-
}

omc-dir(){  echo $(env-home)/$(omc-rel) ; }
omc-sdir(){ echo $(env-home)/$(omc-rel) ; }
omc-tdir(){ echo $(env-home)/$(omc-rel)/tests ; }
omc-idir(){ echo $(local-base)/env/$(omc-rel).install ; }
omc-bdir(){ echo $(local-base)/env/$(omc-rel).build ; }
omc-prefix(){ echo $(omc-idir) ; }


omc-cd(){   cd $(omc-dir); }
omc-scd(){  cd $(omc-sdir); }
omc-tcd(){  cd $(omc-tdir); }
omc-bcd(){  cd $(omc-bdir); }
omc-icd(){  cd $(omc-idir); }

omc-name(){ echo Proj ; }
omc-tag(){  echo PROJ; }

omc-wipe(){ local bdir=$(omc-bdir) ; rm -rf $bdir ; } 
omc-txt(){ vi $(omc-sdir)/CMakeLists.txt $(omc-tdir)/CMakeLists.txt ; } 

omc-cmake(){
   local msg="=== $FUNCNAME : "
   local iwd=$PWD
   local bdir=$(omc-bdir)
   mkdir -p $bdir

   [ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already use omc-cmake-modify to change config or omc-configure to start from scratch  && return  
   omc-bcd

   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(omc-prefix) \
       $* \
       $(omc-sdir)

   cd $iwd 
}


omc-cmake-modify(){
  local msg="=== $FUNCNAME : "
  local bdir=$(omc-bdir)
  local bcache=$bdir/CMakeCache.txt
  [ ! -f "$bcache" ] && echo $msg requires a preexisting $bcache from prior opticks-cmake run && return
  omc-bcd

  cmake \
          . 
}

omc-configure(){
   omc-wipe
   omc-cmake
}
omc-all(){
   omc-configure
   omc--
   omc-ctest
}

omc-sln(){ echo $(omc-bdir)/$(omc-name).sln ; }
omc-slnw(){  vs- ; echo $(vs-wp $(omc-sln)) ; }
omc-vs(){
   vs-
   local sln=$1
   [ -z "$sln" ] && sln=$(omc-sln)
   local slnw=$(vs-wp $sln)

    cat << EOC
# sln  $sln
# slnw $slnw
# copy/paste into powershell v2 
#
devenv /useenv $slnw
EOC

}

#omc-config(){ echo Debug ; }
omc-config(){ echo RelWithDebInfo ; }
omc--(){

   local msg="$FUNCNAME : "
   local iwd=$PWD

   local bdir=$(omc-bdir)
   [ ! -d "$bdir" ] && echo $msg bdir $bdir does not exist && return

   cd $bdir

   cmake --build . --config $(omc-config) --target ${1:-install}

   cd $iwd
}

omc-ctest()
{
   local msg="$FUNCNAME : "
   local iwd=$PWD

   local bdir=$1
   shift
   [ -z "$bdir" ] && bdir=$(omc-bdir)
   cd $bdir

   ctest $*

   cd $iwd
   echo $msg use -V to show output 
}




omc-end-template(){ echo 130 ; }
