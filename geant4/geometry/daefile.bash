# === func-gen- : geant4/geometry/daefile fgp geant4/geometry/daefile.bash fgn daefile fgh geant4/geometry
daefile-src(){      echo geant4/geometry/daefile.bash ; }
daefile-source(){   echo ${BASH_SOURCE:-$(env-home)/$(daefile-src)} ; }
daefile-vi(){       vi $(daefile-source) ; }
daefile-env(){      elocal- ; }
daefile-usage(){ cat << EOU

DAE
====

Try to duplicate VRML2 export under DAE name with extraneous files removed, 
in order to establish baseline to start Collada .dae format export.


EOU
}
daefile-dir(){ echo $(local-base)/env/geant4/geometry/geant4/geometry-daefile ; }
daefile-cd(){  cd $(daefile-dir); }
daefile-mate(){ mate $(daefile-dir) ; }
daefile-get(){
   local dir=$(dirname $(daefile-dir)) &&  mkdir -p $dir && cd $dir

}


daefile-build(){
   cd $(env-home)/geant4/geometry/DAEFILE
   nuwa-
   make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(nuwa-xercesc-idir) G4INSTALL=$(nuwa-g4-bdir)  CPPVERBOSE=1 $1

}


daefile-install(){
   local name=libG4DAEFILE.so
   local blib=$(nuwa-g4-bdir)/lib/Linux-g++/$name
   local ilib=$(nuwa-g4-idir)/lib/$name
   local cmd="cp $blib $ilib"
   echo $cmd
   eval $cmd
   ls -l $blib $ilib
}


