# === func-gen- : tools/civetweb fgp tools/civetweb.bash fgn civetweb fgh tools src base/func.bash
civetweb-source(){   echo ${BASH_SOURCE} ; }
civetweb-edir(){ echo $(dirname $(civetweb-source)) ; }
civetweb-ecd(){  cd $(civetweb-edir); }
civetweb-dir(){  echo $LOCAL_BASE/env/tools/civetweb ; }
civetweb-cd(){   cd $(civetweb-dir); }
civetweb-vi(){   vi $(civetweb-source) ; }
civetweb-env(){  elocal- ; }
civetweb-usage(){ cat << EOU

CivetWeb is an easy to use, powerful, C/C++ embeddable web server with optional CGI, SSL and Lua support.
===========================================================================================================

* https://github.com/civetweb/civetweb/tree/master/docs
* https://civetweb.github.io/civetweb/
* https://civetweb.github.io/civetweb/Embedding.html


Alt
----

* oatpp- ; oatpp-vi

EOU
}
civetweb-get(){
   local dir=$(dirname $(civetweb-dir)) &&  mkdir -p $dir && cd $dir

}
