# === func-gen- : tools/cls fgp tools/cls.bash fgn cls fgh tools
cls-src(){      echo tools/cls.bash ; }
cls-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cls-src)} ; }
cls-vi(){       vi $(cls-source) ; }
cls-env(){      elocal- ; }
cls-usage(){ cat << EOU





EOU
}
cls-dir(){ echo $(local-base)/env/tools/tools-cls ; }
cls-cd(){  cd $(cls-dir); }
cls-mate(){ mate $(cls-dir) ; }
cls-get(){
   local dir=$(dirname $(cls-dir)) &&  mkdir -p $dir && cd $dir

}
