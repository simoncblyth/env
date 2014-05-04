# === func-gen- : zeromq/zeromq fgp zeromq/zeromq.bash fgn zeromq fgh zeromq
zeromq-src(){      echo zeromq/zeromq.bash ; }
zeromq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(zeromq-src)} ; }
zeromq-vi(){       vi $(zeromq-source) ; }
zeromq-env(){      
   elocal- ; 
   export ZEROMQ_PREFIX=$(zeromq-prefix)
}
zeromq-usage(){ cat << EOU

ZEROMQ
======


* http://zguide.zeromq.org/py:all


INSTALLS
-----------

D : as Chroma pyzmq dependency 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ZMQ that came with Chroma/pyzmq is 4.0.3 but where are the headers?::

    delta.local /usr/local/env/chroma_env/lib/python2.7/site-packages/zmq/__init__.pyc 4.0.3 14.0.1

Below headers refernce zmq.h but cannot find it::

    (chroma_env)delta:chroma_env blyth$ find $VIRTUAL_ENV -name zmq*.h
    /usr/local/env/chroma_env/lib/python2.7/site-packages/zmq/utils/zmq_compat.h
    /usr/local/env/chroma_env/lib/python2.7/site-packages/zmq/utils/zmq_constants.h



HELLOWORLD SERVER
-------------------

::

    [blyth@belle7 zeromq_hello_c]$ IPTABLES_PORT=5555 iptables-webopen
    [blyth@belle7 zeromq_hello_c]$ LD_LIBRARY_PATH=$(zeromq-prefix)/lib zeromq-hello-server
    zmq_recv ...... after zmq_recv [hello0]
    zmq_send ...... after zmq_send [hello0]
    zmq_recv ...... after zmq_recv [hello1]
    zmq_send ...... after zmq_send [hello1]
    zmq_recv ...... after zmq_recv [hello2]

    ...
    [blyth@belle7 e]$ IPTABLES_PORT=5555 iptables-webclose




EOU
}
zeromq-fold(){ echo $(local-base)/env/zeromq ; }
zeromq-dir(){ echo $(zeromq-fold)/$(zeromq-name) ; }
zeromq-sdir(){ echo $(env-home)/zeromq ; }
zeromq-cd(){  cd $(zeromq-dir)/$1; }
zeromq-scd(){ cd $(zeromq-sdir)/$1; }
zeromq-mate(){ mate $(zeromq-dir) ; }

zeromq-name(){ echo zeromq-4.0.4 ; }
zeromq-url(){  echo http://download.zeromq.org/$(zeromq-name).tar.gz ; }
zeromq-get(){
    local dir=$(dirname $(zeromq-dir)) &&  mkdir -p $dir && cd $dir
    local url=$(zeromq-url)
    local tgz=$(basename $url)
    local nam=${tgz/.tar.gz}
    echo url $url tgz $tgz nam $nam

    [ ! -f "$tgz" ] && curl -O $url
    [ ! -d "$nam" ] && tar zxvf $tgz
}
zeromq-prefix(){ 
  case $NODE_TAG in 
    D) echo $VIRTUAL_ENV ;; 
    G) echo $(zeromq-fold) ;;
    *) echo $(zeromq-fold) ;;
  esac
}
zeromq-make(){
  zeromq-cd
  ./configure --prefix=$(zeromq-prefix)
  make 
  make install
}




zeromq-zguide-get(){
  git clone --depth=1 git://github.com/imatix/zguide.git
}
zeromq-versions(){
   python -c "import zmq, socket ; print socket.gethostname(), zmq.__file__, zmq.zmq_version(), zmq.pyzmq_version() "
}




zeromq-hello-server-host(){  echo ${HELLO_SERVER_HOST:-belle7.nuu.edu.tw} ; }
zeromq-hello-server-port(){  echo ${HELLO_SERVER_PORT:-5555} ; }

zeromq-hello-config(){
    local host=$(zeromq-hello-server-host)
    local port=$(zeromq-hello-server-port)
    export HELLO_SERVER_CONFIG="tcp://*:$port" 
    export HELLO_CLIENT_CONFIG="tcp://$host:$port" 
    env | grep HELLO
}
zeromq-hello-make(){  cd $(zeromq-sdir)/zeromq_hello_c && ./make.sh ; }
zeromq-hello-server-Darwin(){ zeromq-hello-config ;                                      /tmp/hwserver ; }      ## OSX default linking bakes in path to the lib ?
zeromq-hello-server-Linux(){  zeromq-hello-config ; LD_LIBRARY_PATH=$(zeromq-prefix)/lib /tmp/hwserver ; }   
zeromq-hello-server(){ $FUNCNAME-$(uname) ; }

zeromq-hello-client(){ zeromq-hello-config ; /tmp/hwclient ; }


zeromq-echoserver-make(){
  local iwd=$PWD
  zeromq-scd zeromq_echoserver
  local name=echoserver 
  local bin=/tmp/$name
  cc -I$ZEROMQ_PREFIX/include -c $name.c && cc -L$ZEROMQ_PREFIX/lib -lzmq $name.o -o $bin && rm $name.o 
  ls -l $bin
  cd $iwd
}
zeromq-echoserver-config(){ echo "tcp://*:5555" ; }
zeromq-echoserver-run(){
  LD_LIBRARY_PATH=$ZEROMQ_PREFIX/lib ECHO_SERVER_CONFIG=$(zeromq-echoserver-config) /tmp/echoserver 
}
zeromq-echoserver-gdb(){
  LD_LIBRARY_PATH=$ZEROMQ_PREFIX/lib ECHO_SERVER_CONFIG=$(zeromq-echoserver-config) gdb /tmp/echoserver 
}




