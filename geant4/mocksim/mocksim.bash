# === func-gen- : geant4/mocksim/mocksim fgp geant4/mocksim/mocksim.bash fgn mocksim fgh geant4/mocksim
mocksim-src(){      echo geant4/mocksim/mocksim.bash ; }
mocksim-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mocksim-src)} ; }
mocksim-vi(){       vi $(mocksim-source) ; }
mocksim-usage(){ cat << EOU

Mockup of Geant4 based Simulation App
======================================

Objective 
---------

Fast cycle development/testing of Geant4 level 
code such as G4DAEChroma.

Usage on D or N
-----------------

::

    [blyth@belle7 ~]$ mocksim.sh 
    mocksim : sourced /home/blyth/env-dybx.sh
    G4GDML: Reading '/data1/env/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml'...
    G4GDML: Reading definitions...

    delta:~ blyth$ mocksim.sh 
    mocksim : failed to find /Users/blyth/env-dybx.sh using default environment
    G4GDML: Reading '/usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.gdml'...


Installs
----------

* N: using gmake manual config access to Geant4 installed as NuWa dependency
     **NB needs non-standard NuWa DYBX install with GDML/XercesC/G4DAE enabled**

* D: using cmake config access to Geant4 installed as Chroma dependency

  * http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/ch03s02.html


FUNCTIONS
-----------

*mocksim-configure*
     configure does appropriate thing for detected node

*mocksim-build*
     build does appropriate thing for detected node


EOU
}
mocksim-dir(){ echo $(local-base)/env/geant4/mocksim ; }
mocksim-srcdir(){ echo $(env-home)/geant4/mocksim ; }
mocksim-cd(){  cd $(mocksim-dir); }
mocksim-scd(){  cd $(mocksim-srcdir); }
mocksim-name(){ echo mocksim ; }

mocksim-env(){      
   elocal- 
   [ "$NODE_TAG" == "D" ] && chroma-
}

mocksim-geant4-dir(){
   case $NODE_TAG in
     D) echo $(chroma-geant4-dir) ;;
   esac
}

mocksim-prep(){
   local bdir=$(mocksim-dir)
   mkdir -p $bdir
}

mocksim-configure(){
   mocksim-prep 
   mocksim-cd
   case $NODE_TAG in 
     D) $FUNCNAME-cmake ;;
     *) $FUNCNAME-gmake ;;
   esac  
}
mocksim-configure-gmake(){ echo -n ; }
mocksim-configure-cmake(){ cmake -DGeant4_DIR=$(mocksim-geant4-dir) $(mocksim-srcdir) ; }


mocksim-build(){
   case $NODE_TAG in 
     D) $FUNCNAME-cmake ;;
     *) $FUNCNAME-gmake ;;
   esac  
}

mocksim-build-cmake(){ 
   mocksim-cd
   make 
}



############ all the ugliness of the build is hidden by cmake, but not gmake  

mocksim-build-gmake-env(){
   local fenv="$HOME/env-dybx.sh"    
   [ -f "$fenv" ] && source $fenv
}

mocksim-build-gmake(){

   mocksim-cd
   $FUNCNAME-env

   local srcdir=$(mocksim-srcdir)
   local name=$(mocksim-name)
   local exepath=$name

   gdml-
   # if omit the xercesc incdir the system xerces-c gets used causing linker problems later
   g++ -c -I$(gdml-g4-incdir) \
          -I$(gdml-clhep-incdir) \
          -I$(gdml-xercesc-incdir) \
           -DG4LIB_USE_GDML \
        $srcdir/$name.cc -o $name.o

   #local opt="-m32"
   local opt=""

   g++ $opt $name.o -o $exepath \
        -L$(gdml-xercesc-libdir) -lxerces-c  \
        -L$(gdml-g4-libdir) \
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
           -lG4Tree $(mocksim-build-gmake-vis-libline) \
           -lG4modeling $(mocksim-build-gmake-clhep-libline) \
             -lm

    rm $name.o
}

mocksim-build-gmake-clhep-libline(){
   ## later g4 does not need, internal CLHEP? 
   case $NODE_TAG in 
     N) echo -L$(gdml-clhep-libdir) -l$(gdml-clhep-lib)  ;;
     D) echo -n ;;
   esac
}
mocksim-build-gmake-vis-libline(){
   ## didnt get to work on D
   case $NODE_TAG in 
     N) echo -lG4OpenGL -lG4vis_management ;;
     D) echo -n ;;
   esac
}



