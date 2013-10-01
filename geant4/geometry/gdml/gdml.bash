# === func-gen- : geant4/geometry/gdml/gdml fgp geant4/geometry/gdml/gdml.bash fgn gdml fgh geant4/geometry/gdml
gdml-src(){      echo geant4/geometry/gdml/gdml.bash ; }
gdml-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gdml-src)} ; }
gdml-vi(){       vi $(gdml-source) ; }
gdml-env(){      elocal- ; }
gdml-usage(){ cat << EOU





EOU
}
gdml-dir(){ echo $(local-base)/env/geant4/geometry/gdml/geant4/geometry/gdml-gdml ; }
gdml-cd(){  cd $(gdml-dir); }
gdml-mate(){ mate $(gdml-dir) ; }
gdml-get(){
   local dir=$(dirname $(gdml-dir)) &&  mkdir -p $dir && cd $dir

}

gdml-build(){
   cd $DYB/external/build/LCG/geant4.9.2.p01/source/persistency/gdml
   make CLHEP_BASE_DIR=$DYB/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$DYB/external/XercesC/2.8.0/i686-slc5-gcc41-dbg

}
 


