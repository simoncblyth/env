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


CMake Build
------------

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/ch03s02.html



FUNCTIONS
-----------

mocksim-cmake


EOU
}
mocksim-dir(){ echo $(local-base)/env/geant4/mocksim ; }
mocksim-srcdir(){ echo $(env-home)/geant4/mocksim ; }
mocksim-cd(){  cd $(mocksim-dir); }
mocksim-scd(){  cd $(mocksim-srcdir); }

mocksim-env(){      
   elocal- 
   [ "$NODE_TAG" == "D" ] && chroma-
}

mocksim-geant4-dir(){
   case $NODE_TAG in
     D) echo $(chroma-geant4-dir) ;;
   esac
}

mocksim-cmake(){
  local bdir=$(mocksim-dir)
  mkdir -p $bdir
  cd $bdir 
  cmake -DGeant4_DIR=$(mocksim-geant4-dir) $(mocksim-srcdir)  
}


