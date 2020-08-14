# === func-gen- : embree/ospray fgp embree/ospray.bash fgn ospray fgh embree src base/func.bash
ospray-source(){   echo ${BASH_SOURCE} ; }
ospray-edir(){ echo $(dirname $(ospray-source)) ; }
ospray-ecd(){  cd $(ospray-edir); }
ospray-dir(){  echo $LOCAL_BASE/env/embree/ospray ; }
ospray-cd(){   cd $(ospray-dir); }
ospray-vi(){   vi $(ospray-source) ; }
ospray-env(){  elocal- ; }
ospray-usage(){ cat << EOU

Intel OSPRay
==============


* http://www.ospray.org/downloads.html


EOU
}
ospray-get(){
   local dir=$(dirname $(ospray-dir)) &&  mkdir -p $dir && cd $dir

}
