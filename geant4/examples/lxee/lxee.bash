# === func-gen- : geant4/examples/lxee/lxee fgp geant4/examples/lxee/lxee.bash fgn lxee fgh geant4/examples/lxee
lxee-src(){      echo geant4/examples/lxee/lxee.bash ; }
lxee-source(){   echo ${BASH_SOURCE:-$(env-home)/$(lxee-src)} ; }
lxee-vi(){       vi $(lxee-source) ; }
lxee-env(){      elocal- ; }
lxee-usage(){ cat << EOU





EOU
}
lxee-dir(){ echo $(local-base)/env/geant4/examples/lxee/geant4/examples/lxee-lxee ; }
lxee-cd(){  cd $(lxee-dir); }
lxee-mate(){ mate $(lxee-dir) ; }
lxee-get(){
   local dir=$(dirname $(lxee-dir)) &&  mkdir -p $dir && cd $dir

}
