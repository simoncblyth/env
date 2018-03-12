# === func-gen- : adm/gitsplit fgp adm/gitsplit.bash fgn gitsplit fgh adm
gitsplit-src(){      echo adm/gitsplit.bash ; }
gitsplit-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gitsplit-src)} ; }
gitsplit-vi(){       vi $(gitsplit-source) ; }
gitsplit-env(){      elocal- ; }
gitsplit-usage(){ cat << EOU





EOU
}
gitsplit-dir(){ echo $(local-base)/env/adm/adm-gitsplit ; }
gitsplit-cd(){  cd $(gitsplit-dir); }
gitsplit-mate(){ mate $(gitsplit-dir) ; }
gitsplit-get(){
   local dir=$(dirname $(gitsplit-dir)) &&  mkdir -p $dir && cd $dir

}
