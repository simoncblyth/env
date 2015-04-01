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

Alternatives
-------------

* http://nanomsg.org/documentation-zeromq.html



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


GUI EVENTLOOP INTEGRATION
---------------------------

* :google:`zeromq integrate gui event loop`

http://zeromq.org/area:faq

    The zmq_poll() function accepts a timeout so if you need to poll and process
    GUI events in the same application thread you can set a timeout and
    periodically poll for GUI events. 



ZeroMQ multi threading with PAIR
----------------------------------

::

    delta:C blyth$ zeromq-

    delta:C blyth$ zeromq-examples-get mtrelay.c
    cp /usr/local/env/zeromq/zguide/examples/C/mtrelay.c /Users/blyth/env/zeromq/zguide/examples/C/mtrelay.c

    delta:C blyth$ zeromq-clang mtrelay.c 
    clang -I/usr/local/env/zeromq/include -L/usr/local/env/zeromq/lib -lzmq mtrelay.c -o /tmp/env/zeromq/mtrelay

    delta:C blyth$ /tmp/env/zeromq/mtrelay
    Step 1 ready, signaling step 2
    Step 2 ready, signaling step 3
    Test successful!





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
zeromq-prefix(){ echo ${ZEROMQ_PREFIX:-$(zeromq-prefix-default)} ;}
zeromq-prefix-default(){ 
  case $NODE_TAG in 
    D_original) echo /usr/local/env/chroma_env ;;   ## happens to be VIRTUAL_ENV 
    D) echo $(zeromq-fold) ;; 
    G) echo $(zeromq-fold) ;;
    *) echo $(zeromq-fold) ;;
  esac
}

zeromq-idir(){ echo $(zeromq-prefix)/include ; }
zeromq-ldir(){ echo $(zeromq-prefix)/lib ; }
zeromq-clang(){ 
   mkdir -p /tmp/env/zeromq
   local cmd="clang -I$(zeromq-idir) -L$(zeromq-ldir) -lzmq $1 -o /tmp/env/zeromq/${1/.c}"
   echo $cmd
   eval $cmd
}


zeromq-make(){
  zeromq-cd
  env-llp

  ./configure --prefix=$(zeromq-prefix)
  make 
  make install
}

zeromq-build(){
   zeromq-get
   zeromq-make
}


zeromq-ls(){ ls $(zeromq-prefix)/include $(zeromq-prefix)/lib ; }

zeromq-zguide-install-zhelpers(){
   #cp $(zeromq-zguide-dir)/examples/C/zhelpers.h $(zeromq-prefix)/include/
   cp $(zeromq-zguide-dir)/examples/C/zhelpers.h $(zmq-dir)/  # better to keep this with sources to avoid extra install step
}


zeromq-zguide-dir(){ echo $(zeromq-fold)/zguide ; }
zeromq-zguide-cd(){ cd $(zeromq-zguide-dir) ; }
zeromq-zguide-get(){
  cd $(dirname $(zeromq-zguide-dir)) 
  [ ! -d zguide ] && git clone git://github.com/imatix/zguide.git
}
zeromq-zguide-find(){
   zeromq-zguide-cd
   find . -name '*.c' -exec grep -l ${1:-czmq} {} \;
}


zeromq-versions(){
   python -c "import zmq, socket ; print socket.gethostname(), zmq.__file__, zmq.zmq_version(), zmq.pyzmq_version() "
}




zeromq-examples-lang(){ echo C ; }
zeromq-examples-idir(){ echo $(zeromq-fold)/zguide/examples/$(zeromq-examples-lang) ; }
zeromq-examples-sdir(){ echo $(env-home)/zeromq/zguide/examples/$(zeromq-examples-lang) ; }
zeromq-examples-icd(){  cd $(zeromq-examples-idir) ; }
zeromq-examples-scd(){  cd $(zeromq-examples-sdir) ; }
zeromq-examples-get(){
   local nam=${1:-mtrelay.c}
   local src=$(zeromq-examples-idir)/$nam ;
   local dst=$(zeromq-examples-sdir)/$nam ;
   mkdir -p $(dirname $dst)
   local cmd="cp $src $dst"
   echo $cmd
   eval $cmd
   zeromq-examples-scd
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


