# === func-gen- : tools/libmicrohttpd fgp tools/libmicrohttpd.bash fgn libmicrohttpd fgh tools src base/func.bash
libmicrohttpd-source(){   echo ${BASH_SOURCE} ; }
libmicrohttpd-edir(){ echo $(dirname $(libmicrohttpd-source)) ; }
libmicrohttpd-ecd(){  cd $(libmicrohttpd-edir); }
libmicrohttpd-dir(){  echo $LOCAL_BASE/env/tools/libmicrohttpd ; }
libmicrohttpd-cd(){   cd $(libmicrohttpd-dir); }
libmicrohttpd-vi(){   vi $(libmicrohttpd-source) ; }
libmicrohttpd-env(){  elocal- ; }
libmicrohttpd-usage(){ cat << EOU


https://www.gnu.org/software/libmicrohttpd/







EOU
}
libmicrohttpd-get(){
   local dir=$(dirname $(libmicrohttpd-dir)) &&  mkdir -p $dir && cd $dir

}
