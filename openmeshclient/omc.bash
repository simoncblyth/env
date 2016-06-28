# === func-gen- : openmeshclient/omc fgp openmeshclient/omc.bash fgn omc fgh openmeshclient
omc-src(){      echo openmeshclient/omc.bash ; }
omc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(omc-src)} ; }
omc-vi(){       vi $(omc-source) ; }
omc-env(){      elocal- ; }
omc-usage(){ cat << EOU





EOU
}
omc-dir(){ echo $(local-base)/env/openmeshclient/openmeshclient-omc ; }
omc-cd(){  cd $(omc-dir); }
omc-mate(){ mate $(omc-dir) ; }
omc-get(){
   local dir=$(dirname $(omc-dir)) &&  mkdir -p $dir && cd $dir

}




