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

   curl --socks5 127.0.0.1:8080  -L -O https://github.com/simoncblyth/opticks/archive/refs/tags/v0.2.3.tar.gz 


   curl --socks5 127.0.0.1:8080  -L -O https://github.com/simoncblyth/opticks/archive/refs/tags/v0.2.0.zip
   curl -x socks5://127.0.0.1:8080  -L -O https://github.com/simoncblyth/opticks/archive/refs/tags/v0.2.0.zip
   ALL_PROXY=socks5://127.0.0.1:8080 curl -L -O https://github.com/simoncblyth/opticks/archive/refs/tags/v0.2.0.zip


   > cat ~/.curlrc 
   proxy=socks5://127.0.0.1:8080

   > curl -L -O https://github.com/simoncblyth/opticks/archive/refs/tags/v0.2.0.zip 

  
   echo "proxy=socks5://127.0.0.1:8080" > ~/.curlrc 

   echo "proxy=socks5h://127.0.0.1:8080" > ~/.curlrc    ## better


socks5 OR socks5h
-------------------

The difference between SOCKS5 and SOCKS5H is where the Domain Name System (DNS)
resolution occurs. With SOCKS5, DNS resolution happens on the client's device
(client-side) before connecting to the proxy. With SOCKS5H, DNS resolution is
handled by the SOCKS server itself, passing the domain name directly to the
server. 



NB in addition to the above command need to first start the proxy
-------------------------------------------------------------------

For example with::

   soks-fg

The nub of that is::

    epsilon:examples blyth$ soks-fg-start-
    ssh -C -ND localhost:8080 B





EOU
}



