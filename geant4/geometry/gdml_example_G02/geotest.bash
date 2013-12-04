# === func-gen- : geant4/geometry/gdml_example_G02/geotest fgp geant4/geometry/gdml_example_G02/geotest.bash fgn geotest fgh geant4/geometry/gdml_example_G02
geotest-src(){      echo geant4/geometry/gdml_example_G02/geotest.bash ; }
geotest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(geotest-src)} ; }
geotest-vi(){       vi $(geotest-source) $* ; }
geotest-env(){      elocal- ; g4- ; nuwa- ; fenv ;  }
geotest-usage(){ cat << EOU

GEOTEST
=========

CAUTION when omitted to setup the NuWa environment got non 
obvious result of failing to load libFR (FukuiRenderer) due to misssing libosc_Coin.so

Link Warning from G4FR::

          -lgeotest   -lG4FR -lG4visHepRep -lG4RayTracer -lG4VRML -lG4Tree
    -lG4vis_management -lG4modeling -lG4interfaces -lG4persistency
    -lG4error_propagation -lG4readout -lG4physicslists -lG4run -lG4event
    -lG4tracking -lG4parmodels -lG4processes -lG4digits_hits -lG4track
    -lG4particles -lG4geometry -lG4materials -lG4graphics_reps -lG4intercoms
    -lG4global
    -L/data1/env/local/dyb/external/XercesC/2.8.0/i686-slc5-gcc41-dbg/lib
    -lxerces-c   -lCLHEP -lm /usr/bin/ld: warning: libCoinXt.so, needed by
    /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/lib/Linux-g++/libG4FR.so,     not found (try using -rpath or -rpath-link)
    /usr/bin/ld: warning: libosc_Coin.so, needed by /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/lib/Linux-g++/libG4FR.so, not found (try using -rpath or -rpath-link)

EOU
}
geotest-name(){ echo gdml_example_G02 ; }
geotest-dir(){ echo $(env-home)/geant4/geometry/$(geotest-name) ; }
geotest-cd(){  cd $(geotest-dir); }
geotest-mate(){ mate $(geotest-dir) ; }
geotest-get(){
   local dir=$(dirname $(geotest-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(geotest-name)
   [ -d $nam ] && echo dir $nam already exists && return 1
   local cmd="cp -R $(g4-dir)/examples/extended/persistency/gdml/G02 $nam"
   echo $cmd
}

geotest-make(){
   geotest-cd
   make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(nuwa-xercesc-idir) G4INSTALL=$(nuwa-g4-bdir) CPPVERBOSE=1  $*
}

geotest-prep(){
   geotest-cd
   local out=wtest.gdml
   [ -f "$out" ] && rm -f $out
}

geotest-exe(){
  echo $(nuwa-g4-xdir)/geotest  
}

geotest-run(){
   geotest-prep
   LIBC_FATAL_STDERR_=1 MALLOC_CHECK_=1 $(geotest-exe) $*
}

geotest-gdb(){
   geotest-prep
   gdb $(geotest-exe) --args $(geotest-exe) $*
}

