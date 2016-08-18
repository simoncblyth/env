# === func-gen- : graphics/collada/collada fgp graphics/collada/collada.bash fgn collada fgh graphics/collada
collada-src(){      echo graphics/collada/collada.bash ; }
collada-source(){   echo ${BASH_SOURCE:-$(env-home)/$(collada-src)} ; }
collada-vi(){       vi $(collada-source) ; }
collada-env(){      elocal- ; }
collada-usage(){ cat << EOU

COLLADA
=========

* https://www.khronos.org/news/press/khronos-collada-now-recognized-as-iso-standard


EOU
}
collada-dir(){ echo $(local-base)/env/graphics/collada ; }
collada-cd(){  cd $(collada-dir); }
collada-mate(){ mate $(collada-dir) ; }
collada-get(){
   local dir=$(dirname $(collada-dir)) &&  mkdir -p $dir && cd $dir

}
