#!/bin/bash -l

arg=$1

do_make(){
    local tgt=$1
    nuwa-
    make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(nuwa-xercesc-idir) G4INSTALL=$(nuwa-g4-bdir) CPPVERBOSE=1  $tgt
}

do_install(){
    echo installing the lib with dae-install
    dae-
    dae-install
}

if [ "$arg" == "clean" ]; then 
    do_make clean
    do_make && do_install
else
    do_make && do_install
fi


