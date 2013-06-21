# === func-gen- : proxy/proxy fgp proxy/proxy.bash fgn proxy fgh proxy
proxy-src(){      echo proxy/proxy.bash ; }
proxy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(proxy-src)} ; }
proxy-vi(){       vi $(proxy-source) ; }
proxy-env(){      elocal- ; }
proxy-usage(){ cat << EOU

PROXY
======

For usage of socks.pac file with Safari.app  see :doc:`/bash/ssh`


EOU
}
proxy-dir(){ echo $(local-base)/env/proxy/proxy-proxy ; }
proxy-cd(){  cd $(proxy-dir); }
proxy-mate(){ mate $(proxy-dir) ; }
proxy-get(){
   local dir=$(dirname $(proxy-dir)) &&  mkdir -p $dir && cd $dir

}
