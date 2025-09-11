# === func-gen- : tools/drogon fgp tools/drogon.bash fgn drogon fgh tools src base/func.bash
drogon-source(){   echo ${BASH_SOURCE} ; }
drogon-edir(){ echo $(dirname $(drogon-source)) ; }
drogon-ecd(){  cd $(drogon-edir); }
drogon-dir(){  echo $LOCAL_BASE/env/tools/drogon ; }
drogon-cd(){   cd $(drogon-dir); }
drogon-vi(){   vi $(drogon-source) ; }
drogon-env(){  elocal- ; }
drogon-usage(){ cat << EOU


https://github.com/drogonframework/drogon

https://drogonframework.github.io/drogon-docs/#/ENG/ENG-02-Installation






EOU
}
drogon-get(){
   local dir=$(dirname $(drogon-dir)) &&  mkdir -p $dir && cd $dir

}
