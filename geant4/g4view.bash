# === func-gen- : geant4/g4view fgp geant4/g4view.bash fgn g4view fgh geant4
g4view-src(){      echo geant4/g4view.bash ; }
g4view-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4view-src)} ; }
g4view-vi(){       vi $(g4view-source) ; }
g4view-env(){      elocal- ; }
g4view-usage(){ cat << EOU

G4VIEW
=======

Lightweight GDML + Event viewer from Guy Barrand including iOS, Android.

* http://softinex.lal.in2p3.fr/



EOU
}
g4view-dir(){ echo $(local-base)/env/geant4/geant4-g4view ; }
g4view-cd(){  cd $(g4view-dir); }
g4view-mate(){ mate $(g4view-dir) ; }
g4view-get(){
   local dir=$(dirname $(g4view-dir)) &&  mkdir -p $dir && cd $dir

}
