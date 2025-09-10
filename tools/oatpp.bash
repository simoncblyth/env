# === func-gen- : tools/oatpp fgp tools/oatpp.bash fgn oatpp fgh tools src base/func.bash
oatpp-source(){   echo ${BASH_SOURCE} ; }
oatpp-edir(){ echo $(dirname $(oatpp-source)) ; }
oatpp-ecd(){  cd $(oatpp-edir); }
oatpp-dir(){  echo $LOCAL_BASE/env/tools/oatpp ; }
oatpp-cd(){   cd $(oatpp-dir); }
oatpp-vi(){   vi $(oatpp-source) ; }
oatpp-env(){  elocal- ; }
oatpp-usage(){ cat << EOU


* https://github.com/oatpp/oatpp
* https://oatpp.io/
* https://news.ycombinator.com/item?id=18210397
* https://github.com/oatpp/example-crud





EOU
}
oatpp-get(){
   local dir=$(dirname $(oatpp-dir)) &&  mkdir -p $dir && cd $dir

}
