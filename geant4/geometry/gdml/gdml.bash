# === func-gen- : geant4/geometry/gdml/gdml fgp geant4/geometry/gdml/gdml.bash fgn gdml fgh geant4/geometry/gdml
gdml-src(){      echo geant4/geometry/gdml/gdml.bash ; }
gdml-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gdml-src)} ; }
gdml-vi(){       vi $(gdml-source) ; }
gdml-usage(){ cat << EOU


g++ -m32 -Wl,-rpath,/data/env/local/env/boost/boost_1_54_0.local/lib:/data/env/local/dyb/trunk/external/geant4/4.9.2.p01/i686-slc4-gcc34-dbg/lib:/data/env/local/dyb/trunk/external/clhep/2.0.4.2/i686-slc4-gcc34-dbg/lib: -Wl,-soname,G4global.so -shared -o G4global.so  G4PyCoutDestination.o  pyG4ApplicationState.o  pyG4Exception.o  pyG4ExceptionHandler.o  pyG4ExceptionSeverity.o  pyG4RandomDirection.o  pyG4RotationMatrix.o  pyG4StateManager.o  pyG4String.o  pyG4ThreeVector.o  pyG4Timer.o  pyG4Transform3D.o  pyG4TwoVector.o  pyG4UnitsTable.o  pyG4UserLimits.o  pyG4Version.o  pygeomdefs.o  pyglobals.o  pymodG4global.o  pyRandomEngines.o  pyRandomize.o  -L/data/env/local/env/boost/boost_1_54_0.local/lib -lboost_python -L/data/env/local/dyb/trunk/external/XercesC/2.8.0/i686-slc4-gcc34-dbg/lib -lxerces-c -L/data/env/local/dyb/trunk/external/geant4/4.9.2.p01/i686-slc4-gcc34-dbg/lib -lG4persistency -lG4readout -lG4run -lG4event -lG4tracking -lG4parmodels -lG4processes -lG4digits_hits -lG4track -lG4particles -lG4geometry -lG4materials -lG4graphics_reps -lG4intercoms -lG4interfaces -lG4global -lG4physicslists  -lG4FR -lG4visHepRep -lG4RayTracer -lG4VRML -lG4Tree -lG4OpenGL -lG4vis_management -lG4modeling -L/data/env/local/dyb/trunk/external/clhep/2.0.4.2/i686-slc4-gcc34-dbg/lib -lCLHEP-2.0.4.2


System xerces-c causing problems::

    [blyth@belle7 G01]$ rpm -ql xerces-c
    /usr/lib/libxerces-c.so.27
    /usr/lib/libxerces-c.so.27.0
    /usr/lib/libxerces-depdom.so.27
    /usr/lib/libxerces-depdom.so.27.0
    /usr/share/doc/xerces-c-2.7.0
    /usr/share/doc/xerces-c-2.7.0/LICENSE.txt

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
   cp include/* ../../../include/

   # no install target 
   #make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(nuwa-xercesc-idir) install

}

gdml-install-persistency(){
   cd $(nuwa-g4-bdir)/source/persistency
   cp ../../lib/Linux-g++/libG4persistency.so $(nuwa-g4-libdir)/
}


gdml-test(){
   type $FUNCNAME
   cd $(env-home)/geant4/geometry/gdml

   # if omit the xercesc incdir the system xerces-c gets used causing linker problems later
   g++ -c -I$(nuwa-g4-incdir) \
          -I$(nuwa-clhep-incdir) \
          -I$(nuwa-xercesc-incdir) \
           -DG4LIB_USE_GDML \
        gdmltest.cc -o gdmltest.o

   g++ -m32 gdmltest.o -o gdmltest \
        -L$(nuwa-xercesc-libdir) -lxerces-c  \
        -L$(nuwa-g4-libdir) \
           -lG4persistency \
           -lG4readout \
           -lG4run \
           -lG4event \
           -lG4tracking \
           -lG4parmodels \
           -lG4processes \
           -lG4digits_hits \
           -lG4track \
           -lG4particles \
           -lG4geometry -lG4materials -lG4graphics_reps -lG4intercoms \
           -lG4interfaces -lG4global -lG4physicslists  \
           -lG4FR \
           -lG4visHepRep \
           -lG4RayTracer \
           -lG4VRML \
           -lG4Tree \
           -lG4OpenGL \
           -lG4vis_management \
           -lG4modeling \
       -L$(nuwa-clhep-libdir) -l$(nuwa-clhep-lib) -lm


}

gdml-example-g01(){
    cd $(nuwa-g4-bdir)/examples/extended/persistency/gdml/G01
    make CLHEP_BASE_DIR=$(nuwa-clhep-idir)  G4SYSTEM=Linux-g++ XERCESCROOT=$(nuwa-xercesc-idir) CPPVERBOSE=1 G4INSTALL=../../../../.. 
}



