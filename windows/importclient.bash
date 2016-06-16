# === func-gen- : windows/importclient fgp windows/importclient.bash fgn importclient fgh windows
importclient-src(){      echo windows/importclient.bash ; }
importclient-source(){   echo ${BASH_SOURCE:-$(env-home)/$(importclient-src)} ; }
importclient-vi(){       vi $(importclient-source) ; }
importclient-env(){      elocal- ; }
importclient-usage(){ cat << EOU





EOU
}
importclient-dir(){ echo $(local-base)/env/windows/windows-importclient ; }
importclient-cd(){  cd $(importclient-dir); }
importclient-mate(){ mate $(importclient-dir) ; }
importclient-get(){
   local dir=$(dirname $(importclient-dir)) &&  mkdir -p $dir && cd $dir

}
