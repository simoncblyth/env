# === func-gen- : geant4/g4op/g4opgen fgp geant4/g4op/g4opgen.bash fgn g4opgen fgh geant4/g4op
g4opgen-src(){      echo geant4/g4op/g4opgen.bash ; }
g4opgen-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4opgen-src)} ; }
g4opgen-vi(){       vi $(g4opgen-source) ; }
g4opgen-usage(){ cat << EOU

Geant4 Optical Photon Generation Notes
========================================
      

* https://en.wikipedia.org/wiki/Birks%27_Law


EOU
}
g4opgen-env(){      elocal- ; g4- ; }
g4opgen-dir(){ echo $(local-base)/env/geant4/g4op/geant4/g4op-g4opgen ; }
g4opgen-cd(){  cd $(g4opgen-dir); }
g4opgen-mate(){ mate $(g4opgen-dir) ; }
g4opgen-get(){
   local dir=$(dirname $(g4opgen-dir)) &&  mkdir -p $dir && cd $dir

}


g4opgen-paths(){ cat << EOP
source/processes/electromagnetic/xrays/include/G4Cerenkov.hh
source/processes/electromagnetic/xrays/src/G4Cerenkov.cc
source/processes/electromagnetic/xrays/include/G4Scintillation.hh
source/processes/electromagnetic/xrays/src/G4Scintillation.cc
source/processes/electromagnetic/utils/include/G4EmSaturation.hh
source/processes/electromagnetic/utils/src/G4EmSaturation.cc
EOP
}

g4opgen-edit(){  g4-cd ; vi $(g4opgen-paths) ; }



