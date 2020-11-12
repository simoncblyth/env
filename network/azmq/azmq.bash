# === func-gen- : network/azmq/azmq fgp network/azmq/azmq.bash fgn azmq fgh network/azmq
azmq-src(){      echo network/azmq/azmq.bash ; }
azmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(azmq-src)} ; }
azmq-vi(){       vi $(azmq-source) ; }
azmq-env(){      elocal- ; }
azmq-usage(){ cat << EOU

AZMQ
======


* https://github.com/zeromq/azmq


The azmq library provides Boost Asio style bindings for ZeroMQ

This library is built on top of ZeroMQ's standard C interface and is intended
to work well with C++ applications which use the Boost libraries in general,
and Asio in particular.

The main abstraction exposed by the library is azmq::socket which provides an
Asio style socket interface to the underlying zeromq socket and interfaces with
Asio's io_service(). The socket implementation participates in the io_service's
reactor for asynchronous IO and may be freely mixed with other Asio socket
types (raw TCP/UDP/Serial/etc.).


Library dependencies are -

Boost 1.48 or later
ZeroMQ 4.0.x



Integrating Boost Asio with ZeroMQ, 24 Dec 2014
-------------------------------------------------

* https://rodgert.github.io/2014/12/24/boost-asio-and-zeromq-pt1/




EOU
}
azmq-edir(){ echo $(env-home)/network/azmq; }
azmq-ecd(){  cd $(azmq-edir) ; }

azmq-name(){ echo azmq ; }
#azmq-dir(){ echo $(local-base)/env/network/azmq ; }
azmq-dir(){   echo $(azmq-prefix).build/$(azmq-name) ; }  # exploded distribution dir
azmq-prefix(){ echo ${OPTICKS_AZMQ_PREFIX:-$(opticks-prefix)_externals/$(azmq-name)}  ; }

azmq-cd(){  cd $(azmq-dir); }
azmq-bcd(){  cd $(azmq-bdir); }

azmq-sdir(){ echo $(azmq-dir) ; }
azmq-bdir(){ echo $(azmq-dir).build ; }


#azmq-url(){  echo https://github.com/zeromq/azmq.git ; }
azmq-url(){  echo git://github.com/simoncblyth/azmq.git ; }


azmq-info(){ cat << EOI

  azmq-prefix : $(azmq-prefix) 
  azmq-dir    : $(azmq-dir)

EOI
}

azmq-get(){
   local dir=$(dirname $(azmq-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d azmq ] && git clone $(azmq-url)
}

azmq-wipe(){
   local bdir=$(azmq-bdir)  ;
   rm -rf $bdir
}
azmq-cmake(){
   local iwd=$PWD
   local sdir=$(azmq-sdir) ;
   local bdir=$(azmq-bdir)  ;
   mkdir -p $bdir
   azmq-bcd
   cmake $sdir -DCMAKE_INSTALL_PREFIX=$(azmq-prefix) 
   cd $iwd
}

azmq-make(){
   azmq-bcd

   make
   make test
   make install
}

azmq--(){
  azmq-get
  azmq-cmake
  azmq-make
}
  






azmq-test-url(){ echo https://github.com/onier/testAzmq ; }
azmq-test-dir(){ echo $LOCAL_BASE/env/network/azmq-test/testAzmq ; }
azmq-test-cd(){  cd $(azmq-test-dir) ; }
azmq-test-get()
{
   local name=$(basename $(azmq-test-dir))
   local fold=$(dirname $(azmq-test-dir))
   [ ! -d "$fold" ] && mkdir -p $fold 
   [ ! -d "$fold" ] && echo $msg FAIL to create $fold && return 1
 
   cd $fold
   [ ! -d "$name" ] && git clone $(azmq-test-url)
}

