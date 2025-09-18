# === func-gen- : tools/drogon fgp tools/drogon.bash fgn drogon fgh tools src base/func.bash
drogon-source(){   echo ${BASH_SOURCE} ; }
drogon-edir(){ echo $(dirname $(drogon-source)) ; }
drogon-ecd(){  cd $(drogon-edir); }
drogon-dir(){  echo $LOCAL_BASE/env/tools/drogon ; }
drogon-cd(){   cd $(drogon-dir); }
drogon-vi(){   vi $(drogon-source) ; }
drogon-env(){  elocal- ; }
drogon-usage(){ cat << EOU
Drogon C++ web framework
=========================

https://github.com/drogonframework/drogon

https://drogonframework.github.io/drogon-docs/#/

https://drogonframework.github.io/drogon-docs/#/ENG/ENG-02-Installation


macOS install deps
--------------------

* https://drogonframework.github.io/drogon-docs/#/ENG/ENG-02-Installation

::

    zeta:env blyth$ sudo port search jsoncpp
    jsoncpp @1.9.5 (devel)
        JSON C++ library

    zeta:env blyth$ sudo port search ossp-uuid
    ossp-uuid @1.6.2_13 (devel)
        ISO-C API and CLI for generating Universally Unique Identifiers



macOS install
---------------


::

    zeta:env blyth$ pwd
    /usr/local/env

    zeta:env blyth$ git clone https://github.com/drogonframework/drogon
    Cloning into 'drogon'...
    remote: Enumerating objects: 20060, done.
    remote: Counting objects: 100% (2703/2703), done.
    remote: Compressing objects: 100% (436/436), done.
    remote: Total 20060 (delta 2502), reused 2268 (delta 2267), pack-reused 17357 (from 2)
    Receiving objects: 100% (20060/20060), 6.14 MiB | 2.34 MiB/s, done.
    Resolving deltas: 100% (13510/13510), done.

    zeta:env blyth$ cd drogon
    zeta:drogon blyth$ which git
    /usr/bin/git
    zeta:drogon blyth$ git submodule update --init
    Submodule 'trantor' (https://github.com/an-tao/trantor.git) registered for path 'trantor'
    Cloning into '/usr/local/env/drogon/trantor'...
    Submodule path 'trantor': checked out '43fd79b2dbac59608a819ebba167e8fe2c079d90'
    zeta:drogon blyth$ 

    zeta:drogon blyth$ mkdir build
    zeta:drogon blyth$ cd build
    zeta:build blyth$ cmake -DCMAKE_INSTALL_PREFIX=/usr/local/e .. 


macOS /usr/local/env/drogon_check
-----------------------------------

::

    zeta:env blyth$ drogon-set-path
    zeta:env blyth$ which drogon_ctl
    /usr/local/e/bin/drogon_ctl

    zeta:env blyth$ drogon_ctl create project drogon_check
    create a project named drogon_check
    zeta:env blyth$ l drogon_check/
    total 112
     8 -rw-r--r--   1 blyth  staff   2683 Sep 18 15:52 CMakeLists.txt
     8 -rw-r--r--   1 blyth  staff    375 Sep 18 15:52 main.cc
    24 -rw-r--r--   1 blyth  staff   9632 Sep 18 15:52 .gitignore
    40 -rw-r--r--   1 blyth  staff  17922 Sep 18 15:52 config.json
    32 -rw-r--r--   1 blyth  staff  14915 Sep 18 15:52 config.yaml
     0 drwxr-xr-x   4 blyth  staff    128 Sep 18 15:52 test
     0 drwxr-xr-x   3 blyth  staff     96 Sep 18 15:52 models
     0 drwxr-xr-x  14 blyth  staff    448 Sep 18 15:52 .
     0 drwxr-xr-x   2 blyth  staff     64 Sep 18 15:52 build
     0 drwxr-xr-x   2 blyth  staff     64 Sep 18 15:52 plugins
     0 drwxr-xr-x   2 blyth  staff     64 Sep 18 15:52 filters
     0 drwxr-xr-x   2 blyth  staff     64 Sep 18 15:52 controllers
     0 drwxr-xr-x   2 blyth  staff     64 Sep 18 15:52 views
     0 drwxr-xr-x   8 blyth  staff    256 Sep 18 15:52 ..
    zeta:env blyth$ 


Build the generated project (see build.log for full output)::

    zeta:build blyth$ cmake -DCMAKE_INSTALL_PREFIX=/usr/local/e  ..
    zeta:build blyth$ make


::

    zeta:build blyth$ cat run.sh 
    #!/bin/bash

    usage(){ cat << EOU
    run.sh
    =======

    ::

       open -a Safari.app http://127.0.0.1:5555


    ::

        zeta:build blyth$ lsof -i :5555
        COMMAND     PID  USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
        drogon_ch 51218 blyth   13u  IPv4 0x72ce034c6a1dda8f      0t0  TCP *:personal-agent (LISTEN)
        zeta:build blyth$ 


    CAUTION: without PATH, DYLD_LIBRARY_PATH setup no error is reported, nothing is logged and no html is served


Try adding controller
-----------------------

::

    zeta:drogon_check blyth$ which drogon_ctl
    /usr/local/e/bin/drogon_ctl
    zeta:drogon_check blyth$ drogon_ctl create controller HelloCtrl
    Create a http simple controller: HelloCtrl

::

    curl http://127.0.0.1:5555/test/   # gives 404 until explicitly added that path
    curl http://127.0.0.1:5555/test    # OK





EOU
}

PATH=/usr/local/e/bin:$PATH
DYLD_LIBRARY_PATH=/usr/local/e/lib:$DYLD_LIBRARY_PATH

./drogon_check 




multiPart form data and handling large uploads
------------------------------------------------

* https://drogonframework.github.io/drogon-docs/#/ENG/ENG-09-1-File-Handler
* https://github.com/drogonframework/drogon/issues/1974
* https://clehaxze.tw/gemlog/2022/07-09-handle-large-file-upload-with-drogon-web-framework.gmi

cf with django https://www.django-rest-framework.org/api-guide/parsers/


* https://drogonframework.github.io/drogon-docs/#/ENG/ENG-09-1-File-Handler






EOU
}
drogon-get(){
   local dir=$(dirname $(drogon-dir)) &&  mkdir -p $dir && cd $dir

}


drogon-set-path(){

   PATH=/usr/local/e/bin:$PATH
   DYLD_LIBRARY_PATH=/usr/local/e/lib:$DYLD_LIBRARY_PATH

}

