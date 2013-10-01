# === func-gen- : geant4/geometry/gdml/gdml fgp geant4/geometry/gdml/gdml.bash fgn gdml fgh geant4/geometry/gdml
gdml-src(){      echo geant4/geometry/gdml/gdml.bash ; }
gdml-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gdml-src)} ; }
gdml-vi(){       vi $(gdml-source) ; }
gdml-usage(){ cat << EOU





EOU
}
gdml-env(){      
   elocal- 
   nuwa-
}
gdml-dir(){ echo $(nuwa-g4-bdir)/source/persistency/gdml ; }
gdml-sdir(){ echo $(env-home)/geant4/geometry/gdml ; }
gdml-cd(){  cd $(gdml-dir); }
gdml-scd(){  cd $(gdml-sdir); }
gdml-mate(){ mate $(gdml-dir) ; }
gdml-get(){
   local dir=$(dirname $(gdml-dir)) &&  mkdir -p $dir && cd $dir

}

gdml-build(){
   cd $(nuwa-g4-bdir)/source/persistency/gdml
   make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(nuwa-xercesc-idir)
}

gdml-build-persistency(){
   cd $(nuwa-g4-bdir)/source/persistency
   make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(nuwa-xercesc-idir) 
   make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(nuwa-xercesc-idir) global
}
 
gdml-install(){
   cd $(nuwa-g4-bdir)/source/persistency/gdml
   cp ../../../lib/Linux-g++/libG4gdml.so $(nuwa-g4-libdir)/
   cp include/* $(nuwa-g4-incdir)/

   # no install target 
   #make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(nuwa-xercesc-idir) install

}

gdml-install-persistency(){
   cd $(nuwa-g4-bdir)/source/persistency
   cp ../../lib/Linux-g++/libG4persistency.so $(nuwa-g4-libdir)/
}


