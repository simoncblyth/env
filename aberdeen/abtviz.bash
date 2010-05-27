# === func-gen- : aberdeen/abtviz fgp aberdeen/abtviz.bash fgn abtviz fgh aberdeen
abtviz-src(){      echo aberdeen/abtviz.bash ; }
abtviz-source(){   echo ${BASH_SOURCE:-$(env-home)/$(abtviz-src)} ; }
abtviz-vi(){       vi $(abtviz-source) ; }
abtviz-env(){      elocal- ; }
abtviz-usage(){
  cat << EOU
     abtviz-src : $(abtviz-src)
     abtviz-dir : $(abtviz-dir)


EOU
}
abtviz-dir(){ echo $(local-base)/env/aberdeen/aberdeen-abtviz ; }
abtviz-cd(){  cd $(abtviz-dir); }
abtviz-mate(){ mate $(abtviz-dir) ; }
abtviz-get(){
   local dir=$(dirname $(abtviz-dir)) &&  mkdir -p $dir && cd $dir

}
