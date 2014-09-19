# === func-gen- : chroma/ChromaZMQRootTest/czrt fgp chroma/ChromaZMQRootTest/czrt.bash fgn czrt fgh chroma/ChromaZMQRootTest
czrt-src(){      echo chroma/ChromaZMQRootTest/czrt.bash ; }
czrt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(czrt-src)} ; }
czrt-vi(){       vi $(czrt-source) ; }
czrt-env(){      elocal- ; }
czrt-usage(){ cat << EOU

CZRT : ChromaZMQRootTest 
===========================

#. TODO: bring echoserver code into czrt folder, its already in the nuwapkg 

Memory Issue
--------------

With MyTMessage used on stack in the test (rather than heap) 
get **pointer being freed was not allocated**
as it goes out of scope::

    ChromaZMQRootTest(36786,0x7fff7ab15310) malloc: *** error for object 0x10561df08: pointer being freed was not allocated
    *** set a breakpoint in malloc_error_break to debug

Suspect something funny related to the TObject status bits and
the fact that the object is serialized in one place and deserialised elsewhere.
Saw something similar with pyzmq which was resolved using PyROOT `ROOT.SetOwnnership(kFALSE)`.

Currently MyTMessage usage on heap without deletion looks like a leak.
Need to investigate TObject internals to understand further.

Running Echoserver
-------------------

::

    [blyth@belle7 cmt]$ czrt-nuwapkg-testserver
    === nuwacmt-config : for pkg /data1/env/local/dyb/NuWa-trunk/dybgaudi/Utilities/ChromaZMQRootTest
    Removing all previous make fragments from i686-slc5-gcc41-dbg
    Creating setup scripts.
    Creating cleanup scripts.
    ZMQEchoServer.exe
    do_bind tcp://*:5555 

FUNCTIONS
-----------

*czrt-build*
    cmake controlled build  

*czrt-build-full*
    build after first deleting the build directory

*czrt-nuwapkg-cpto*
    copy source into the corresponding DYB NuWa pkg


SCRIPTS
--------

*czrt.sh*
    send test CPL TObject to the configured *zmq-broker-url-frontend*
 

Local DEBUG
-------------

#. Initially *czrt.sh* gives symbol loading issue 
#. After a *czrt-build* runs, but meets NUU network blockage::

    delta:collada blyth$ czrt.sh

       zmq-broker-url          : tcp://203.64.184.126:5001
       zmq-broker-url-frontend : tcp://203.64.184.126:5001
       zmq-broker-url-backend  : tcp://203.64.184.126:5002

       zmq-broker-host         : 203.64.184.126

    ZMQRoot::ZMQRoot envvar [CHROMA_CLIENT_CONFIG][Q] config [tcp://203.64.184.126:5001] 
    ChromaPhotonList::Print  [6]
    ZMQRoot::SendObject sent bytes: 457 


#. Attempt to tunnel through the block, getting refused::

    delta:~ blyth$ czrt.sh --zmqtunnelnode=N

       zmq-broker-url          : tcp://203.64.184.126:5001
       zmq-broker-url-frontend : tcp://203.64.184.126:5001
       zmq-broker-url-backend  : tcp://203.64.184.126:5002

       zmq-broker-host         : 203.64.184.126

    === ssh-tunnel-open : opening ssh tunnel using below command
    ssh -fN -p 22 -L 127.0.0.1:64542:203.64.184.126:5001 N
    === ssh-tunnel-open : modifying ZMQ_BROKER_URL_FRONTEND to tcp://127.0.0.1:64542 in order to use the ssh tunnel
    ZMQRoot::ZMQRoot envvar [CHROMA_CLIENT_CONFIG][Q] config [tcp://127.0.0.1:64542] 
    ChromaPhotonList::Print  [6]
    ZMQRoot::SendObject sent bytes: 457 
    channel 1: open failed: connect failed: Connection refused
    channel 1: open failed: connect failed: Connection refused
    channel 1: open failed: connect failed: Connection refused



NuWa DEBUG
-----------

Not getting through, nothing showing on broker::

    [blyth@belle7 env]$ LD_LIBRARY_PATH=$DYB/external/zmq/4.0.4/i686-slc5-gcc41-dbg/lib:$LD_LIBRARY_PATH CHROMA_CLIENT_CONFIG=tcp://127.0.0.1:5001 ChromaZMQRootTest.exe
    ZMQRoot::ZMQRoot envvar [CHROMA_CLIENT_CONFIG] config [tcp://127.0.0.1:5001] 
    ChromaPhotonList::Print UID [0] [6]
    ZMQRoot::SendObject sent bytes: 457 


works locally
~~~~~~~~~~~~~~

::

    [blyth@belle7 src]$ LD_LIBRARY_PATH=$DYB/external/zmq/4.0.4/i686-slc5-gcc41-dbg/lib:$LD_LIBRARY_PATH ECHO_SERVER_CONFIG=tcp://*:4000 ZMQEchoServer.exe
    do_bind tcp://*:4000 
    do_receive zmq_msg_recv waiting... 
    do_receive got bytes:457 
    do_send... sending msg of  457 bytes 
    do_send... queued  457 bytes 
    do_receive zmq_msg_recv waiting... 

    [blyth@belle7 ~]$ LD_LIBRARY_PATH=$DYB/external/zmq/4.0.4/i686-slc5-gcc41-dbg/lib:$LD_LIBRARY_PATH CHROMA_CLIENT_CONFIG=tcp://127.0.0.1:4000 ChromaZMQRootTest.exe
    ZMQRoot::ZMQRoot envvar [CHROMA_CLIENT_CONFIG] config [tcp://127.0.0.1:4000] 
    ChromaPhotonList::Print UID [0] [6]
    ZMQRoot::SendObject sent bytes: 457 
    ZMQRoot::ReceiveObject received bytes: 457 
    ZMQRoot::ReceiveObject reading TObject from the TMessage 
    ZMQRoot::ReceiveObject returning TObject 
    ReceiveObject
    ChromaPhotonList::Print UID [0] [6]
    ChromaPhotonList::Print UID [0] [6]
    ChromaPhotonList::Details [6]
     index 0 pos (1,1,1) mom (2,2,2) pol (3,3,3) _t 0 _wavelength 100 _pmtid 101
     index 1 pos (1,1,1) mom (2,2,2) pol (3,3,3) _t 0 _wavelength 100 _pmtid 101
     index 2 pos (1,1,1) mom (2,2,2) pol (3,3,3) _t 0 _wavelength 100 _pmtid 101
     index 3 pos (1,1,1) mom (2,2,2) pol (3,3,3) _t 0 _wavelength 100 _pmtid 101
     index 4 pos (1,1,1) mom (2,2,2) pol (3,3,3) _t 0 _wavelength 100 _pmtid 101
     index 5 pos (1,1,1) mom (2,2,2) pol (3,3,3) _t 0 _wavelength 100 _pmtid 101
    [blyth@belle7 ~]$ 




EOU
}
czrt-cd(){  cd $(czrt-dir); }
czrt-mate(){ mate $(czrt-dir) ; }
czrt-get(){
   local dir=$(dirname $(czrt-dir)) &&  mkdir -p $dir && cd $dir

}
czrt-name(){ echo ChromaZMQRootTest ; }
czrt-dir(){  echo $(local-base)/env/chroma/$(czrt-name) ; }
czrt-sdir(){ echo $(env-home)/chroma/$(czrt-name) ; }
czrt-bdir(){ echo /tmp/env/chroma/$(czrt-name) ; }

czrt-bin(){ echo $(czrt-prefix)/bin/$(czrt-name) ; }

czrt-cd(){   cd $(czrt-sdir); }
czrt-icd(){  cd $(czrt-dir); }
czrt-scd(){  cd $(czrt-sdir); }
czrt-bcd(){  cd $(czrt-bdir); }

czrt-verbose(){ echo 1 ; }
czrt-prefix(){ echo $(czrt-dir) ; }

czrt-geant4-home(){ 
  case $NODE_TAG in 
    D) echo /usr/local/env/chroma_env/src/geant4.9.5.p01 ;;
  esac
}
czrt-geant4-dir(){ 
  case $NODE_TAG in 
    D) echo /usr/local/env/chroma_env/lib/Geant4-9.5.1 ;;
  esac
}
czrt-rootsys(){
  case $NODE_TAG in 
    D) echo /usr/local/env/chroma_env/src/root-v5.34.14 ;;
  esac
}


czrt-env(){      
   elocal- 
   export GEANT4_HOME=$(czrt-geant4-home)
   export ROOTSYS=$(czrt-rootsys)   # needed to find rootcint for dictionary creation
}


czrt-wipe(){
   local msg="=== $FUNCNAME :"
   local bdir="$(czrt-bdir)"
   echo $msg deleting bdir $bdir
   rm -rf "$bdir"
}
czrt-cmake(){
   type $FUNCNAME
   local iwd=$PWD
   mkdir -p $(czrt-bdir)   
   czrt-bcd
   cmake  \
         -DCMAKE_INSTALL_PREFIX=$(czrt-prefix) \
         $(czrt-sdir) 

   cd $iwd
}
czrt-make(){
   local iwd=$PWD
   czrt-bcd
   make $* VERBOSE=$(czrt-verbose) 
   cd $iwd
}
czrt-install(){
   czrt-make install
}
czrt-build(){
   czrt-cmake
   czrt-make
   czrt-install
}
czrt-build-full(){
   czrt-wipe
   czrt-build
}
czrt-otool(){
   otool-
   otool-info $(czrt-lib)
}

czrt-nuwapkg(){    echo $DYB/NuWa-trunk/dybgaudi/Utilities/$(czrt-name) ; }  
czrt-nuwapkg-cd(){ cd $(czrt-nuwapkg)/$1 ; }
czrt-nuwapkg-cpto(){

   local pkg=$(czrt-nuwapkg)   
   local nam=$(basename $pkg)
   local inc=$pkg/$nam
   local src=$pkg/src

   mkdir -p $pkg
   mkdir -p $pkg/src
  
   local iwd=$PWD 
   czrt-scd

   local main=ChromaZMQRootTest.cc
   cp $main    $src/

   perl -pi -e 's,ChromaPhotonList.hh,Chroma/ChromaPhotonList.hh,' $src/$main
   perl -pi -e 's,ZMQRoot.hh,ZMQRoot/ZMQRoot.hh,'                  $src/$main


   cd $iwd
}

czrt-nuwapkg-env(){
   local iwd=$PWD

   fenv            # implicit assumption that fast env matches the DYB-installation
   nuwacmt-
   nuwacmt-config $(czrt-nuwapkg)

   cd $iwd 
}

czrt-nuwapkg-build(){

   czrt-nuwapkg-env 
   czrt-nuwapkg-cd cmt
   
   cmt make  
}


czrt-nuwapkg-testserver(){ echoserver- ; czrt-nuwapkg-run $(echoserver-name) $* ; }
czrt-nuwapkg-testclient(){               czrt-nuwapkg-run $(czrt-name)       $* ; }
czrt-nuwapkg-run(){
   local app=$1
   shift
   czrt-nuwapkg-env
   local cmd="$* $app.exe"    # allow environment override 
   echo $cmd
   eval $cmd
}

czrt-nsend(){
  [ "$NODE_TAG" != "N" ] && echo $msg this is for debug use on N only && return
  fenv
  LD_LIBRARY_PATH=$DYB/external/zmq/4.0.4/$CMTCONFIG/lib:$LD_LIBRARY_PATH CHROMA_CLIENT_CONFIG=tcp://127.0.0.1:5001 ChromaZMQRootTest.exe
}

