curl-vi(){ vi $BASH_SOURCE ; }
curl-env(){ echo -n ; }
curl-usage(){ cat << EOU
curl.bash
===========

* https://curl.se/docs/tutorial.html
* https://reqbin.com/req/c-ddxflki5/curl-proxy-server


Download zip using socks5 proxy
---------------------------------

::

   curl --socks5 127.0.0.1:8080  -L -O https://github.com/simoncblyth/opticks/archive/refs/tags/v0.2.0.zip
   curl -x socks5://127.0.0.1:8080  -L -O https://github.com/simoncblyth/opticks/archive/refs/tags/v0.2.0.zip
   ALL_PROXY=socks5://127.0.0.1:8080 curl -L -O https://github.com/simoncblyth/opticks/archive/refs/tags/v0.2.0.zip


   > cat ~/.curlrc 
   proxy=socks5://127.0.0.1:8080

   > curl -L -O https://github.com/simoncblyth/opticks/archive/refs/tags/v0.2.0.zip 


EOU
}



