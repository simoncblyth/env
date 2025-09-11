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

alt
----

* https://github.com/Stiffstream/restinio
* https://stiffstream.com/en/docs/restinio/0.7/


RESTinio oatpp civetweb

* https://www.reddit.com/r/cpp/comments/cjj9t5/what_c_web_server_library_one_should_use_nowadays/
* https://www.reddit.com/r/cpp/comments/12653pg/rest_apis_using_c_is_this_even_done_much/
* https://www.reddit.com/r/cpp/comments/17nt8uc/experience_using_crow_as_web_server/


* https://github.com/uNetworking/uWebSockets

EOU
}
oatpp-get(){
   local dir=$(dirname $(oatpp-dir)) &&  mkdir -p $dir && cd $dir

}
