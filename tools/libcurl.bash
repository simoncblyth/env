# === func-gen- : tools/libcurl fgp tools/libcurl.bash fgn libcurl fgh tools src base/func.bash
libcurl-source(){   echo ${BASH_SOURCE} ; }
libcurl-edir(){ echo $(dirname $(libcurl-source)) ; }
libcurl-ecd(){  cd $(libcurl-edir); }
libcurl-dir(){  echo $LOCAL_BASE/env/tools/libcurl ; }
libcurl-cd(){   cd $(libcurl-dir); }
libcurl-vi(){   vi $(libcurl-source) ; }
libcurl-env(){  elocal- ; }
libcurl-usage(){ cat << EOU


A[blyth@localhost env]$ dnf list installed | grep curl
curl.x86_64                                      7.76.1-31.el9_6.1                  @baseos              
libcurl.x86_64                                   7.76.1-31.el9_6.1                  @baseos              
libcurl-devel.x86_64                             7.76.1-31.el9_6.1                  @appstream           
python3-pycurl.x86_64                            7.43.0.6-8.el9                     @AppStream           
A[blyth@localhost env]$ 

A[blyth@localhost env]$ rpm -ql libcurl
/usr/lib/.build-id
/usr/lib/.build-id/b4
/usr/lib/.build-id/b4/01fc87dd99d24239baaa0f4228664be40e87fe
/usr/lib64/libcurl.so.4
/usr/lib64/libcurl.so.4.7.0
/usr/share/licenses/libcurl
/usr/share/licenses/libcurl/COPYING

[blyth@localhost env]$ rpm -ql libcurl-devel
/usr/bin/curl-config
/usr/include/curl
/usr/include/curl/curl.h
/usr/include/curl/curlver.h
/usr/include/curl/easy.h
/usr/include/curl/mprintf.h
/usr/include/curl/multi.h
/usr/include/curl/options.h
/usr/include/curl/stdcheaders.h
/usr/include/curl/system.h
/usr/include/curl/typecheck-gcc.h
/usr/include/curl/urlapi.h
/usr/lib64/libcurl.so
/usr/lib64/pkgconfig/libcurl.pc
/usr/share/aclocal/libcurl.m4
/usr/share/doc/libcurl-devel
/usr/share/doc/libcurl-devel/10-at-a-time.c
/usr/share/doc/libcurl-devel/ABI.md
/usr/share/doc/libcurl-devel/CONTRIBUTE.md
/usr/share/doc/libcurl-devel/INTERNALS.md
/usr/share/doc/libcurl-devel/Makefile.example
/usr/share/doc/libcurl-devel/altsvc.c
/usr/share/doc/libcurl-devel/anyauthput.c
/usr/share/doc/libcurl-devel/cacertinmem.c

A[blyth@localhost env]$ curl-config --version
libcurl 7.76.1








EOU
}
libcurl-get(){
   local dir=$(dirname $(libcurl-dir)) &&  mkdir -p $dir && cd $dir

}
