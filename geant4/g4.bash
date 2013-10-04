# === func-gen- : geant4/g4 fgp geant4/g4.bash fgn g4 fgh geant4
g4-src(){      echo geant4/g4.bash ; }
g4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4-src)} ; }
g4-vi(){       vi $(g4-source) ; }
g4-env(){      elocal- ; nuwa- ;  }
g4-usage(){ cat << EOU





EOU
}
g4-dir(){ echo $(nuwa-g4-bdir); }
g4-cd(){  cd $(g4-dir)/$1; }
g4-mate(){ mate $(g4-dir) ; }
g4-get(){
   local dir=$(dirname $(g4-dir)) &&  mkdir -p $dir && cd $dir

}
