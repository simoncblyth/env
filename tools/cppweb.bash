# === func-gen- : tools/cppweb fgp tools/cppweb.bash fgn cppweb fgh tools src base/func.bash
cppweb-source(){   echo ${BASH_SOURCE} ; }
cppweb-edir(){ echo $(dirname $(cppweb-source)) ; }
cppweb-ecd(){  cd $(cppweb-edir); }
cppweb-dir(){  echo $LOCAL_BASE/env/tools/cppweb ; }
cppweb-cd(){   cd $(cppweb-dir); }
cppweb-vi(){   vi $(cppweb-source) ; }
cppweb-env(){  elocal- ; }
cppweb-usage(){ cat << EOU


* https://drogon.org/







EOU
}
cppweb-get(){
   local dir=$(dirname $(cppweb-dir)) &&  mkdir -p $dir && cd $dir

}
