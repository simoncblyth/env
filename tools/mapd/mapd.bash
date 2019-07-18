# === func-gen- : tools/mapd/mapd fgp tools/mapd/mapd.bash fgn mapd fgh tools/mapd src base/func.bash
mapd-source(){   echo ${BASH_SOURCE} ; }
mapd-edir(){ echo $(dirname $(mapd-source)) ; }
mapd-ecd(){  cd $(mapd-edir); }
mapd-dir(){  echo $LOCAL_BASE/env/tools/mapd/mapd ; }
mapd-cd(){   cd $(mapd-dir); }
mapd-vi(){   vi $(mapd-source) ; }
mapd-env(){  elocal- ; }
mapd-usage(){ cat << EOU


OmniSciDB (formerly MapD Core) 

* https://www.omnisci.com
* https://github.com/omnisci/omniscidb


EOU
}
mapd-get(){
   local dir=$(dirname $(mapd-dir)) &&  mkdir -p $dir && cd $dir

}
