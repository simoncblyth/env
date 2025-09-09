# === func-gen- : tools/libcpr fgp tools/libcpr.bash fgn libcpr fgh tools src base/func.bash
libcpr-source(){   echo ${BASH_SOURCE} ; }
libcpr-edir(){ echo $(dirname $(libcpr-source)) ; }
libcpr-ecd(){  cd $(libcpr-edir); }
libcpr-dir(){  echo $LOCAL_BASE/env/tools/libcpr ; }
libcpr-cd(){   cd $(libcpr-dir); }
libcpr-vi(){   vi $(libcpr-source) ; }
libcpr-env(){  elocal- ; }
libcpr-usage(){ cat << EOU

C++ Requests: Curl for People  : (C++17 wrapper for libcurl)
===============================================================

* https://github.com/libcpr/cpr

* Darwin install : to /usr/local/e/ from build dir /usr/local/env/cpr see /usr/local/env/cpr/build/build.log::

    cd /usr/local/env
    git clone https://github.com/libcpr/cpr.git
    cd cpr
    mkdir build && cd build
    cmake .. -DCPR_USE_SYSTEM_CURL=ON

    cmake --build . --parallel
    [sudo] cmake --install . --prefix /usr/local/e



EOU
}
libcpr-get(){
   local dir=$(dirname $(libcpr-dir)) &&  mkdir -p $dir && cd $dir

}
