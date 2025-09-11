# === func-gen- : tools/crow fgp tools/crow.bash fgn crow fgh tools src base/func.bash
crow-source(){   echo ${BASH_SOURCE} ; }
crow-edir(){ echo $(dirname $(crow-source)) ; }
crow-ecd(){  cd $(crow-edir); }
crow-dir(){  echo $LOCAL_BASE/env/tools/crow ; }
crow-cd(){   cd $(crow-dir); }
crow-vi(){   vi $(crow-source) ; }
crow-env(){  elocal- ; }
crow-usage(){ cat << EOU


https://github.com/CrowCpp/Crow

https://github.com/CrowCpp/Crow/blob/master/examples/example_file_upload.cpp




EOU
}
crow-get(){
   local dir=$(dirname $(crow-dir)) &&  mkdir -p $dir && cd $dir

}
