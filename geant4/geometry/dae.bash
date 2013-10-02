# === func-gen- : geant4/geometry/dae fgp geant4/geometry/dae.bash fgn dae fgh geant4/geometry
dae-src(){      echo geant4/geometry/dae.bash ; }
dae-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dae-src)} ; }
dae-vi(){       vi $(dae-source) ; }
dae-env(){      elocal- ; }
dae-usage(){ cat << EOU

DAE
====

Try to duplicate VRML2 export under DAE name with extraneous files removed, 
in order to establish baseline to start Collada .dae format export.


EOU
}
dae-dir(){ echo $(local-base)/env/geant4/geometry/geant4/geometry-dae ; }
dae-cd(){  cd $(dae-dir); }
dae-mate(){ mate $(dae-dir) ; }
dae-get(){
   local dir=$(dirname $(dae-dir)) &&  mkdir -p $dir && cd $dir

}


dae-build(){
   cd $(env-home)/geant4/geometry/DAE
   nuwa-
   make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(nuwa-xercesc-idir) G4INSTALL=$(nuwa-g4-bdir)  CPPVERBOSE=1

}


dae-install(){
   local name=libG4DAE.so
   local blib=$(nuwa-g4-bdir)/lib/Linux-g++/$name
   local ilib=$(nuwa-g4-idir)/lib/$name
   local cmd="cp $blib $ilib"
   echo $cmd
   eval $cmd
   ls -l $blib $ilib
}


