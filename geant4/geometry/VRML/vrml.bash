# === func-gen- : geant4/geometry/VRML/vrml fgp geant4/geometry/VRML/vrml.bash fgn vrml fgh geant4/geometry/VRML
vrml-src(){      echo geant4/geometry/VRML/vrml.bash ; }
vrml-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vrml-src)} ; }
vrml-vi(){       vi $(vrml-source) ; }
vrml-usage(){ cat << EOU

VRML
=====

Precision fix and debugging/comparing Geant4 VRML exporter with G4DAE exporter.

FUNCTIONS
----------

*vrml-update*
         does deploy and make

*vrml-deploy*
         copies modified geant4 sources from env working copy into Geant4 source tree

*vrml-make*
         builds and installs library 


EOU
}
vrml-sdir(){ echo $(env-home)/geant4/geometry/VRML ; }
vrml-scd(){  cd $(vrml-sdir); }
vrml-dir(){ echo $(vrml-target) ; }
vrml-cd(){  cd $(vrml-dir) ; }

vrml-env(){      elocal- ; nuwa- ; }
vrml-home(){   echo $ENV_HOME/geant4/geometry/VRML ; }
vrml-target(){ echo $(nuwa-g4-bdir)/source/visualization/VRML ; }
vrml-mode(){   echo ${VRML_MODE:-diff} ; }
vrml-libname(){ echo libG4VRML.so ; }

vrml-paths-(){ cat << EOP
GNUmakefile
src/G4VRML2SceneHandlerFunc.icc
src/G4VRML2FileSceneHandler.cc
include/G4VRML2FileSceneHandler.hh
EOP
}

vrml-deploy(){
  local msg="=== $FUNCNAME :"
  local home=$(vrml-home)
  local target=$(vrml-target)
  local mode=$(vrml-mode)
  local path
  local cmd
  vrml-paths- | while read path ; do 
     cmd="$mode $home/$path $target/$path"
     echo $msg $cmd 
     eval $cmd
  done
}

vrml-update(){
   VRML_MODE="diff" vrml-deploy
   VRML_MODE="cp"   vrml-deploy
   vrml-make
}

vrml-ls(){
  vrml-cd
  local name=$(vrml-libname)
  local lib=../../../lib/Linux-g++/$name
  ls -l $lib $(nuwa-g4-libdir)/$name
}

vrml-make(){
  local msg="=== $FUNCNAME :" 
  type $FUNCNAME

  vrml-cd

  local name=$(vrml-libname)
  local lib=../../../lib/Linux-g++/$name
  echo $msg before
  ls -l $lib $(nuwa-g4-libdir)/$name

  export VERBOSE=1
  rm -f $lib && vrml-make- && cp $lib $(nuwa-g4-libdir)/

  echo $msg after
  ls -l $lib $(nuwa-g4-libdir)/$name
}


vrml-make-(){
  make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 $*
}


